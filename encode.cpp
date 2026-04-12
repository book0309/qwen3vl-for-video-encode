// ============================================================================
// 视频编码器：接收 Python 生成的 QP 图和 YUV420 帧数据，使用宏块级码率分配进行 x264 编码
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// 检查是否有 x264 库支持，如果有则启用完整编码功能，否则降级为日志输出
#if __has_include(<x264.h>)
#include <x264.h>
#define HAVE_X264 1
#else
#define HAVE_X264 0
#endif

namespace {

// 二进制包协议的魔数，用于标识每个帧包的开始
constexpr char kMagic[4] = {'Q', 'P', 'K', '1'};

// H.264 宏块大小（16x16 像素），用于宏块级码率分配
constexpr std::uint32_t kMbSize = 16;

// ============================================================================
// 通用二进制读取辅助函数
// ============================================================================

/// 读取单个 POD 类型数据
/// @param in        输入流
/// @param value     输出变量（引用）
/// @return         成功返回 true，否则 false
template <typename T>
bool readPod(std::istream& in, T& value)
{
	return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

/// 读取指定大小的字节缓冲区
/// @param in        输入流
/// @param buffer    输出缓冲区（将被 resize）
/// @param size      要读取的字节数
/// @return         成功返回 true，否则 false
bool readExact(std::istream& in, std::vector<std::uint8_t>& buffer, std::uint32_t size)
{
	buffer.resize(size);
	return size == 0 || static_cast<bool>(in.read(reinterpret_cast<char*>(buffer.data()), size));
}

/// 计算 QP 值集合的平均值
/// @param qp   QP 数据数组
/// @return     平均 QP 值，空数组返回 23.0（默认值）
double averageQp(const std::vector<std::uint8_t>& qp)
{
	if (qp.empty()) {
		return 23.0;
	}

	std::uint64_t sum = 0;
	for (std::uint8_t v : qp) {
		sum += v;
	}
	return static_cast<double>(sum) / static_cast<double>(qp.size());
}

// ============================================================================
// 帧数据包结构
// ============================================================================

/// Python 端发送的帧数据包格式
/// 包含原始视频帧（YUV420）和对应的宏块级 QP 图
struct FramePacket
{
	std::uint32_t frameId;      ///< 帧序号（从 0 开始）
	std::uint32_t width;         ///< 视频帧的宽度（像素）
	std::uint32_t height;        ///< 视频帧的高度（像素）
	std::uint32_t qpWidth;       ///< QP 图的宽度（单位：块/映射点）
	std::uint32_t qpHeight;      ///< QP 图的高度（单位：块/映射点）
	std::vector<std::uint8_t> y; ///< Y 平面（亮度），大小 = width * height
	std::vector<std::uint8_t> u; ///< U 平面（色度），大小 = width/2 * height/2
	std::vector<std::uint8_t> v; ///< V 平面（色度），大小 = width/2 * height/2
	std::vector<std::uint8_t> qp;///< QP 图，大小 = qpWidth * qpHeight，值范围 [0, 51]
};

// ============================================================================
// 二进制包读取函数
// ============================================================================

/// 从输入流读取一个完整的帧数据包
/// 包格式：[magic(4)][version(4)][frameId][width][height][qpWidth][qpHeight]
///         [ySize][uSize][vSize][qpSize][y_data][u_data][v_data][qp_data]
///
/// @param in      输入流（通常是 stdin 或文件）
/// @param packet  输出的帧数据包（所有字段将被填充）
/// @return        成功返回 true，流尾返回 false，格式错误抛异常
bool readFramePacket(std::istream& in, FramePacket& packet)
{
	// 读取并验证包头的魔数（4 字节）
	// 魔数 "QPK1" 用来同步和校验包的开始位置
	char magic[4]{};
	if (!in.read(magic, 4)) {
		return false;
	}

	if (std::memcmp(magic, kMagic, 4) != 0) {
		throw std::runtime_error("invalid packet magic");
	}

	// 读取包头元数据（共 40 字节）
	std::uint32_t version = 0;
	std::uint32_t ySize = 0;
	std::uint32_t uSize = 0;
	std::uint32_t vSize = 0;
	std::uint32_t qpSize = 0;

	if (!readPod(in, version) || !readPod(in, packet.frameId) || !readPod(in, packet.width) || 
		!readPod(in, packet.height) || !readPod(in, packet.qpWidth) || !readPod(in, packet.qpHeight) || 
		!readPod(in, ySize) || !readPod(in, uSize) || !readPod(in, vSize) || !readPod(in, qpSize)) {
		throw std::runtime_error("unexpected end of stream while reading packet header");
	}

	// 版本检查，目前只支持版本 1
	if (version != 1) {
		throw std::runtime_error("unsupported packet version");
	}

	if (packet.width == 0 || packet.height == 0) {
		throw std::runtime_error("invalid frame size in packet header");
	}

	if ((packet.width % 2) != 0 || (packet.height % 2) != 0) {
		throw std::runtime_error("I420 requires even width/height in packet header");
	}

	const std::uint64_t expectedY = static_cast<std::uint64_t>(packet.width) * packet.height;
	const std::uint64_t expectedU = expectedY / 4;
	const std::uint64_t expectedV = expectedY / 4;
	const std::uint64_t expectedQp = static_cast<std::uint64_t>(packet.qpWidth) * packet.qpHeight;

	if (ySize != expectedY || uSize != expectedU || vSize != expectedV) {
		std::ostringstream oss;
		oss << "invalid plane sizes in packet header, frame=" << packet.frameId
			<< " expected(y,u,v)=" << expectedY << ',' << expectedU << ',' << expectedV
			<< " got=" << ySize << ',' << uSize << ',' << vSize;
		throw std::runtime_error(oss.str());
	}

	if (qpSize != expectedQp) {
		std::ostringstream oss;
		oss << "invalid qp size in packet header, frame=" << packet.frameId
			<< " expected=" << expectedQp << " got=" << qpSize
			<< " qpWidth=" << packet.qpWidth << " qpHeight=" << packet.qpHeight;
		throw std::runtime_error(oss.str());
	}

	// 依次读取 Y、U、V、QP 数据
	if (!readExact(in, packet.y, ySize) || !readExact(in, packet.u, uSize) || 
		!readExact(in, packet.v, vSize) || !readExact(in, packet.qp, qpSize)) {
		throw std::runtime_error("unexpected end of stream while reading packet payload");
	}

	return true;
}

// ============================================================================
// 数据包读取器（流封装）
// ============================================================================

/// 简单的包读取器，逐帧从输入流中读取数据包
class PacketReader
{
public:
	/// 构造函数
	/// @param input  输入流引用
	explicit PacketReader(std::istream& input) : input_(input) {}

	/// 读取下一个数据包
	/// @param packet  输出的帧数据包
	/// @return       成功（读到数据）返回 true，流尾返回 false
	bool next(FramePacket& packet)
	{
		return readFramePacket(input_, packet);
	}

private:
	std::istream& input_;  ///< 输入流引用
};

#if HAVE_X264
// ============================================================================
// x264 编码器（完整版：支持宏块级码率分配）
// ============================================================================

/// x264 视频编码器封装类
/// 
/// 核心功能：
/// 1. 初始化 x264 编码器（预设：veryfast + zero-latency）
/// 2. 将 Python 传入 QP 图映射为“每宏块绝对 QP”
/// 3. 逐帧编码并输出 H.264 比特流
///
/// QP 使用策略：
/// - 按 16x16 宏块划分视频帧
/// - 将 Python 生成的 QP 图映射到每个宏块
/// - 直接使用该 QP 作为宏块目标 QP（不再做平均基准 + 偏移）
class X264Encoder
{
public:
	/// 析构函数：自动关闭编码器并释放资源
	~X264Encoder()
	{
		close();
	}

	/// 初始化编码器
	/// 
	/// @param width   视频帧的宽度（像素）
	/// @param height  视频帧的高度（像素）
	/// @param output  输出流（接收 H.264 比特流）
	/// @return       成功返回 true，失败返回 false
	///
	/// 初始化步骤：
	/// 1. 应用 x264 快速预设 ("veryfast")
	/// 2. 配置输入为 I420（YUV420 平面）格式
	/// 3. 设置零延迟模式（VFR=0, repeat_headers=1）
	/// 4. 关键帧间隔设置为 30 帧
	/// 5. 禁用 B 帧（简化宏块控制）
	/// 6. 使用恒定质量模式 (CQP)
	bool open(std::uint32_t width, std::uint32_t height, std::ostream& output)
	{
		close();

		output_ = &output;
		std::memset(&param_, 0, sizeof(param_));

		// 使用快速预设初始化参数（权衡速度和质量）
		if (x264_param_default_preset(&param_, "veryfast", "zerolatency") < 0) {
			return false;
		}

		// 配置编码参数
		param_.i_csp = X264_CSP_I420;              // 输入颜色空间：YUV420 平面格式
		param_.i_width = static_cast<int>(width);   // 帧宽度
		param_.i_height = static_cast<int>(height); // 帧高度
		param_.i_fps_num = 30;                      // 帧率分子（30 fps）
		param_.i_fps_den = 1;                       // 帧率分母
		param_.i_keyint_max = 30;                   // 最大关键帧间隔
		param_.b_intra_refresh = 1;                 // 启用帧内刷新（提高抗丢包能力）
		param_.b_vfr_input = 0;                     // 禁用变帧率（保证时间同步）
		param_.b_repeat_headers = 1;                // 重复 SPS/PPS（便于接收端同步）
		param_.b_annexb = 1;                        // 使用 Annex B NAL 格式（h.264 标准格式）
		param_.i_threads = 1;                       // 单线程编码（便于确定性和调试）
		param_.i_bframe = 0;                        // 禁用 B 帧（简化宏块级控制）
		param_.rc.i_rc_method = X264_RC_CQP;       // 码率控制：恒定质量（允许宏块级控制）
		param_.rc.i_qp_constant = 23;              // 全局默认 QP

		// 打开编码器
		encoder_ = x264_encoder_open(&param_);
		if (!encoder_) {
			return false;
		}

		// 分配图像缓冲（x264 内部使用）
		if (x264_picture_alloc(&picture_, X264_CSP_I420, param_.i_width, param_.i_height) < 0) {
			close();
			return false;
		}

		return true;
	}

	/// 编码单个帧
	/// 
	/// @param packet  输入的帧数据包（含 YUV420 和 QP 图）
	/// @return       成功返回 true，失败返回 false
	///
	/// 编码流程：
	/// 1. 将 YUV420 数据复制到 x264 图像缓冲
  /// 2. 调用 buildMacroblockAbsoluteQp 将 QP 图转换为宏块绝对 QP
	/// 3. 将绝对 QP 写入 x264 的宏块量化数组
	/// 4. 调用 x264 编码器进行编码
	/// 5. 将输出的 NAL 单元写入输出流
	bool encode(const FramePacket& packet)
	{
		if (!encoder_) {
			return false;
		}

		// 验证 YUV420 数据大小
		// 对于 YUV420 平面格式：Y = W*H，U = W/2*H/2，V = W/2*H/2
		const std::size_t expectedY = static_cast<std::size_t>(packet.width) * static_cast<std::size_t>(packet.height);
		const std::size_t expectedU = expectedY / 4;
		const std::size_t expectedV = expectedY / 4;
		if (packet.y.size() != expectedY || packet.u.size() != expectedU || packet.v.size() != expectedV) {
			std::ostringstream oss;
			oss << "unexpected frame size for I420 input, frame=" << packet.frameId
				<< " expected(y,u,v)=" << expectedY << ',' << expectedU << ',' << expectedV
				<< " got=" << packet.y.size() << ',' << packet.u.size() << ',' << packet.v.size();
			throw std::runtime_error(oss.str());
		}

		// 复制 YUV 数据到 x264 的图像缓冲
		std::memcpy(picture_.img.plane[0], packet.y.data(), packet.y.size());
		std::memcpy(picture_.img.plane[1], packet.u.data(), packet.u.size());
		std::memcpy(picture_.img.plane[2], packet.v.data(), packet.v.size());

        // 直接使用 Python 传入的 QP：这里不再使用“平均基准 QP + 偏移”策略。
		// 对于当前实现，设定基准 QP=0，然后把每个宏块的绝对 QP 写入 quant_offsets。
		// 最终每个宏块目标 QP ≈ 0 + quant_offsets[i]。
		buildMacroblockAbsoluteQp(packet);
		picture_.i_qpplus1 = 1;

		// x264 将读取该数组进行宏块级量化控制。
		picture_.prop.quant_offsets = mbAbsoluteQp_.empty() ? nullptr : mbAbsoluteQp_.data();
		picture_.prop.quant_offsets_free = nullptr;

		// 设置演示时间戳（PTS）
		picture_.i_pts = static_cast<int64_t>(frameIndex_++);

		// 调用 x264 编码器进行编码
		x264_nal_t* nals = nullptr;     // 输出 NAL 单元数组指针
		int nalCount = 0;                // NAL 单元数量
		x264_picture_t picOut{};         // 输出图像信息（用于 B 帧重排）
		const int bytes = x264_encoder_encode(encoder_, &nals, &nalCount, &picture_, &picOut);

		if (bytes < 0) {
			return false;
		}

		// 遍历所有输出的 NAL 单元，写入输出流
		for (int i = 0; i < nalCount; ++i) {
			output_->write(reinterpret_cast<const char*>(nals[i].p_payload), 
						   static_cast<std::streamsize>(nals[i].i_payload));
		}

		return true;
	}

	/// 关闭编码器并释放资源
	void close()
	{
		if (encoder_) {
			// 清空宏块 QP 偏移指针（重要：防止野指针）
			picture_.prop.quant_offsets = nullptr;

			// 释放 x264 内部的图像缓冲
			x264_picture_clean(&picture_);

			// 关闭编码器并释放内存
			x264_encoder_close(encoder_);
			encoder_ = nullptr;
		}
	}

private:
  /// 将 QP 图转换为“每宏块绝对 QP”数组
	/// 
	/// 映射策略：
	/// 1. 计算帧的宏块网格大小（基于 16x16）
	/// 2. 对每个宏块，将其映射到 QP 图坐标（支持任意分辨率）
    /// 3. 取出对应位置的 QP 值，约束到 [0, 51]
	/// 4. 直接写入宏块 QP 数组（不计算平均基准）
	///
	/// @param packet  输入的帧数据包
  void buildMacroblockAbsoluteQp(const FramePacket& packet)
	{
		// 计算宏块网格大小（向上取整）
		const std::uint32_t mbWidth = (packet.width + kMbSize - 1) / kMbSize;
		const std::uint32_t mbHeight = (packet.height + kMbSize - 1) / kMbSize;
		const std::size_t mbCount = static_cast<std::size_t>(mbWidth) * static_cast<std::size_t>(mbHeight);

        // 如果 QP 图为空或尺寸无效，所有宏块默认使用 QP 23。
		if (packet.qp.empty() || packet.qpWidth == 0 || packet.qpHeight == 0 || mbCount == 0) {
           mbAbsoluteQp_.assign(mbCount, 23.0F);
			return;
		}

       mbAbsoluteQp_.assign(mbCount, 23.0F);

		// 遍历所有宏块
		for (std::uint32_t my = 0; my < mbHeight; ++my) {
			for (std::uint32_t mx = 0; mx < mbWidth; ++mx) {
				// 将宏块索引映射到 QP 图坐标
				// 支持 QP 图分辨率与帧分辨率不同的情况
				// 使用最近邻（向下取整）映射：qp_coord = (mb_coord * qp_size) / mb_size
				const std::uint32_t qx = std::min((mx * packet.qpWidth) / mbWidth, packet.qpWidth - 1);
				const std::uint32_t qy = std::min((my * packet.qpHeight) / mbHeight, packet.qpHeight - 1);

				// 计算 QP 图中的线性索引
				const std::size_t qIndex = static_cast<std::size_t>(qy) * packet.qpWidth + qx;

				// 计算宏块网格中的线性索引
				const std::size_t mbIndex = static_cast<std::size_t>(my) * mbWidth + mx;

              // 读取对应位置的 QP 值（越界时默认为 23）
				const std::uint8_t q = qIndex < packet.qp.size() ? packet.qp[qIndex] : static_cast<std::uint8_t>(23);
              mbAbsoluteQp_[mbIndex] = static_cast<float>(std::clamp<int>(q, 0, 51));
			}
		}
	}

  x264_t* encoder_ = nullptr;            ///< x264 编码器实例指针
	x264_param_t param_{};                 ///< x264 编码器参数
	x264_picture_t picture_{};             ///< x264 图像结构（输入缓冲）
	std::vector<float> mbAbsoluteQp_;      ///< 每个宏块的绝对 QP 数组（来自 Python）
	std::ostream* output_ = nullptr;       ///< 输出比特流的目标流
	int64_t frameIndex_ = 0;               ///< 当前帧序号（用于 PTS）
};

#else
// ============================================================================
// x264 编码器（降级版：无 x264 库时的日志输出模式）
// ============================================================================

/// 当编译环境中没有 x264.h 时使用的降级版本
/// 功能：仅输出帧信息到日志，便于测试数据流和包格式的正确性
class X264Encoder
{
public:
	/// 初始化编码器（降级版：仅记录信息）
	bool open(std::uint32_t, std::uint32_t, std::ostream& output)
	{
		output_ = &output;
		return true;
	}

	/// 编码单个帧（降级版：输出日志）
	/// 
	/// 输出格式：frame=<ID> width=<W> height=<H> qp_avg=<QP>
	bool encode(const FramePacket& packet)
	{
		(*output_) << "frame=" << packet.frameId
				   << " width=" << packet.width
				   << " height=" << packet.height
				   << " qp_avg=" << averageQp(packet.qp)
				   << '\n';
		return true;
	}

private:
	std::ostream* output_;  ///< 输出流（通常为标准输出或文件）
};
#endif

/// 根据编译环境返回默认输出路径
/// @return 如果有 x264 支持返回 "output.h264"，否则返回 "encoder.log"
std::string defaultOutputPath()
{
	return HAVE_X264 ? std::string("output.h264") : std::string("encoder.log");
}

} // namespace

// ============================================================================
// 主程序：处理输入、编码、输出
// ============================================================================

/// 程序入口和主处理流程
/// 
/// 使用方法：
/// 1. 从标准输入读取数据：
///    python_encoder.py | videocode.exe
/// 
/// 2. 从文件读取数据：
///    videocode.exe input.bin output.h264
/// 
/// 3. 默认输出到文件：
///    videocode.exe input.bin
///
/// 命令行参数：
///  argv[1] - 输入文件路径（可选，"-" 或省略表示 stdin）
///  argv[2] - 输出文件路径（可选，默认为 "output.h264" 或 "encoder.log"）
///
/// 处理流程：
/// 1. 打开输入流（文件或 stdin）
/// 2. 打开输出流
/// 3. 读取第一帧以获取分辨率
/// 4. 初始化编码器
/// 5. 循环编码所有帧
/// 6. 关闭编码器并输出结果
int main(int argc, char* argv[])
{
	try {
		// 禁用 C++ 和 C 标准库的 I/O 同步，加快流操作速度
		std::ios::sync_with_stdio(false);

#ifdef _WIN32
		// Windows 下默认 stdin 可能是文本模式，二进制管道会被错误截断。
		// 这里强制切换为二进制模式，避免读取 QP/YUV 包时出现意外 EOF。
		if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
			std::cerr << "failed to switch stdin to binary mode\n";
			return 1;
		}
#endif

		// ====================================================================
		// 第一步：打开输入流
		// ====================================================================

		std::istream* input = &std::cin;  // 默认使用标准输入
		std::ifstream inputFile;          // 文件输入流

		// 如果指定了输入文件（且不是 "-"），则从文件读取
		if (argc >= 2 && std::string(argv[1]) != "-") {
			inputFile.open(argv[1], std::ios::binary);
			if (!inputFile) {
				std::cerr << "failed to open input file\n";
				return 1;
			}
			input = &inputFile;
		}

		// ====================================================================
		// 第二步：打开输出流
		// ====================================================================

		// 输出路径：若指定则使用指定的路径，否则使用默认路径
		std::string outputPath = argc >= 3 ? argv[2] : defaultOutputPath();
		std::ofstream outputFile(outputPath, std::ios::binary);
		if (!outputFile) {
			std::cerr << "failed to open output file\n";
			return 1;
		}

		// ====================================================================
		// 第三步：创建包读取器，读取第一帧来确定分辨率
		// ====================================================================

		PacketReader reader(*input);
		X264Encoder encoder;
		FramePacket packet;

		// 尝试读取第一帧
		if (!reader.next(packet)) {
			// 如果没有数据，直接返回成功（可能是空流）
			return 0;
		}

		// ====================================================================
		// 第四步：使用第一帧的分辨率初始化编码器
		// ====================================================================

		if (!encoder.open(packet.width, packet.height, outputFile)) {
			std::cerr << "failed to initialize encoder\n";
			return 1;
		}

		// ====================================================================
		// 第五步：循环编码所有帧
		// ====================================================================

		do {
			// 数据有效性检查
			if (packet.width == 0 || packet.height == 0) {
				std::cerr << "invalid frame size\n";
				return 1;
			}

			// 编码当前帧
			if (!encoder.encode(packet)) {
				double qpMin = 0.0;
				double qpMax = 0.0;
				double qpAvg = averageQp(packet.qp);
				if (!packet.qp.empty()) {
					auto [minIt, maxIt] = std::minmax_element(packet.qp.begin(), packet.qp.end());
					qpMin = static_cast<double>(*minIt);
					qpMax = static_cast<double>(*maxIt);
				}
				std::cerr << "failed to encode frame " << packet.frameId
						  << " qp[min,max,avg]=" << qpMin << ',' << qpMax << ',' << qpAvg << '\n';
				return 1;
			}

			// 读取下一帧
		} while (reader.next(packet));

		// ====================================================================
		// 第六步：编码器自动析构并关闭
		// （无需显式调用 encoder.close()，析构函数会自动处理）
		// ====================================================================

		return 0;
	} 
	catch (const std::exception& ex) {
		// 捕获所有异常并输出错误信息
		std::cerr << ex.what() << '\n';
		return 1;
	}
}







