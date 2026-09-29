// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "Host/AudioStream.h"
#include "Host.h"

#include "common/Assertions.h"
#include "common/Console.h"
#include "common/Error.h"

#include "fmt/format.h"
#include "IconsFontAwesome.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace
{
	class CoreAudioAudioStream final : public AudioStream
	{
	public:
		CoreAudioAudioStream(u32 sample_rate, const AudioStreamParameters& parameters);
		~CoreAudioAudioStream() override;

		void SetPaused(bool paused) override;

		bool Initialize(const char* device_name, bool stretch_enabled, Error* error);

	private:
		static OSStatus RenderCallback(void* ref_con, AudioUnitRenderActionFlags* flags, const AudioTimeStamp* timestamp,
			UInt32 bus, UInt32 num_frames, AudioBufferList* data);

		void Destroy();

		AudioUnit m_unit = nullptr;
		bool m_initialized = false;
	};
} // namespace

static std::string CFStringToStdString(CFStringRef str)
{
	if (!str)
		return {};

	if (const char* direct = CFStringGetCStringPtr(str, kCFStringEncodingUTF8))
		return direct;

	const CFIndex len = CFStringGetMaximumSizeForEncoding(CFStringGetLength(str), kCFStringEncodingUTF8) + 1;
	std::string ret(static_cast<size_t>(len), '\0');
	if (!CFStringGetCString(str, ret.data(), len, kCFStringEncodingUTF8))
		return {};

	ret.resize(std::strlen(ret.c_str()));
	return ret;
}

static std::string GetDeviceStringProperty(AudioObjectID device, AudioObjectPropertySelector selector)
{
	const AudioObjectPropertyAddress addr = {selector, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
	CFStringRef value = nullptr;
	UInt32 size = sizeof(value);
	if (AudioObjectGetPropertyData(device, &addr, 0, nullptr, &size, &value) != noErr || !value)
		return {};

	std::string ret = CFStringToStdString(value);
	CFRelease(value);
	return ret;
}

static bool DeviceHasOutputChannels(AudioObjectID device)
{
	const AudioObjectPropertyAddress addr = {
		kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
	UInt32 size = 0;
	if (AudioObjectGetPropertyDataSize(device, &addr, 0, nullptr, &size) != noErr || size == 0)
		return false;

	std::unique_ptr<u8[]> storage = std::make_unique<u8[]>(size);
	AudioBufferList* list = reinterpret_cast<AudioBufferList*>(storage.get());
	if (AudioObjectGetPropertyData(device, &addr, 0, nullptr, &size, list) != noErr)
		return false;

	for (UInt32 i = 0; i < list->mNumberBuffers; i++)
	{
		if (list->mBuffers[i].mNumberChannels > 0)
			return true;
	}

	return false;
}

static u32 GetDeviceOutputLatencyFrames(AudioObjectID device)
{
	// Buffer frame size range minimum plus the device's own latency, roughly what the HAL can do at best.
	AudioObjectPropertyAddress addr = {
		kAudioDevicePropertyBufferFrameSizeRange, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
	AudioValueRange range = {};
	UInt32 size = sizeof(range);
	u32 frames = 0;
	if (AudioObjectGetPropertyData(device, &addr, 0, nullptr, &size, &range) == noErr)
		frames += static_cast<u32>(range.mMinimum);

	addr.mSelector = kAudioDevicePropertyLatency;
	UInt32 latency = 0;
	size = sizeof(latency);
	if (AudioObjectGetPropertyData(device, &addr, 0, nullptr, &size, &latency) == noErr)
		frames += latency;

	return frames;
}

static std::vector<AudioObjectID> GetAllDevices()
{
	std::vector<AudioObjectID> ret;
	const AudioObjectPropertyAddress addr = {
		kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
	UInt32 size = 0;
	if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, nullptr, &size) != noErr)
		return ret;

	ret.resize(size / sizeof(AudioObjectID));
	if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &size, ret.data()) != noErr)
		ret.clear();
	else
		ret.resize(size / sizeof(AudioObjectID));

	return ret;
}

static AudioObjectID FindDeviceByUID(const char* uid)
{
	for (const AudioObjectID dev : GetAllDevices())
	{
		if (DeviceHasOutputChannels(dev) && GetDeviceStringProperty(dev, kAudioDevicePropertyDeviceUID) == uid)
			return dev;
	}

	return kAudioObjectUnknown;
}

CoreAudioAudioStream::CoreAudioAudioStream(u32 sample_rate, const AudioStreamParameters& parameters)
	: AudioStream(sample_rate, parameters)
{
}

CoreAudioAudioStream::~CoreAudioAudioStream()
{
	Destroy();
}

void CoreAudioAudioStream::Destroy()
{
	if (!m_unit)
		return;

	AudioOutputUnitStop(m_unit);
	if (m_initialized)
		AudioUnitUninitialize(m_unit);
	AudioComponentInstanceDispose(m_unit);
	m_unit = nullptr;
	m_initialized = false;
}

bool CoreAudioAudioStream::Initialize(const char* device_name, bool stretch_enabled, Error* error)
{
	// Channel order matches the CoreAudio labels we assign below, same ordering as the SDL backend.
	static constexpr const std::array<std::pair<SampleReader, std::array<AudioChannelLabel, MAX_OUTPUT_CHANNELS>>,
		static_cast<size_t>(AudioExpansionMode::Count)>
		channel_setups = {{
			// Disabled
			{&StereoSampleReaderImpl, {kAudioChannelLabel_Left, kAudioChannelLabel_Right}},
			// StereoLFE
			{&SampleReaderImpl<AudioExpansionMode::StereoLFE, READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
				 READ_CHANNEL_LFE>,
				{kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LFEScreen}},
			// Quadraphonic
			{&SampleReaderImpl<AudioExpansionMode::Quadraphonic, READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
				 READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
				{kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LeftSurround,
					kAudioChannelLabel_RightSurround}},
			// QuadraphonicLFE
			{&SampleReaderImpl<AudioExpansionMode::QuadraphonicLFE, READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
				 READ_CHANNEL_LFE, READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
				{kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LFEScreen,
					kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround}},
			// Surround51
			{&SampleReaderImpl<AudioExpansionMode::Surround51, READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
				 READ_CHANNEL_FRONT_CENTER, READ_CHANNEL_LFE, READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
				{kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Center,
					kAudioChannelLabel_LFEScreen, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround}},
			// Surround71
			{&SampleReaderImpl<AudioExpansionMode::Surround71, READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
				 READ_CHANNEL_FRONT_CENTER, READ_CHANNEL_LFE, READ_CHANNEL_SIDE_LEFT, READ_CHANNEL_SIDE_RIGHT,
				 READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
				{kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Center,
					kAudioChannelLabel_LFEScreen, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround,
					kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight}},
		}};

	const auto& setup = channel_setups[static_cast<size_t>(m_parameters.expansion_mode)];

	// HALOutput lets us pick a device; with no device set it follows the system default output.
	AudioComponentDescription desc = {};
	desc.componentType = kAudioUnitType_Output;
	desc.componentSubType = kAudioUnitSubType_HALOutput;
	desc.componentManufacturer = kAudioUnitManufacturer_Apple;

	AudioComponent component = AudioComponentFindNext(nullptr, &desc);
	if (!component)
	{
		Error::SetStringView(error, "Could not find HAL output audio component.");
		return false;
	}

	OSStatus status = AudioComponentInstanceNew(component, &m_unit);
	if (status != noErr)
	{
		Error::SetStringFmt(error, "AudioComponentInstanceNew() failed: {}", static_cast<s32>(status));
		m_unit = nullptr;
		return false;
	}

	AudioObjectID device = kAudioObjectUnknown;
	if (device_name && *device_name)
	{
		device = FindDeviceByUID(device_name);
		if (device == kAudioObjectUnknown)
		{
			Host::AddIconOSDMessage("AudioDeviceUnavailable", ICON_FA_VOLUME_HIGH,
				fmt::format("Requested audio output device '{}' not found, using default.", device_name),
				Host::OSD_ERROR_DURATION);
		}
	}
	if (device == kAudioObjectUnknown)
	{
		const AudioObjectPropertyAddress addr = {kAudioHardwarePropertyDefaultOutputDevice,
			kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
		UInt32 size = sizeof(device);
		if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &size, &device) != noErr)
			device = kAudioObjectUnknown;
	}

	if (device != kAudioObjectUnknown)
	{
		status = AudioUnitSetProperty(m_unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
			&device, sizeof(device));
		if (status != noErr)
			WARNING_LOG("Failed to set CoreAudio output device: {}", static_cast<s32>(status));
		else
			INFO_LOG("Using CoreAudio output device '{}'.", GetDeviceStringProperty(device, kAudioObjectPropertyName));
	}

	AudioStreamBasicDescription format = {};
	format.mSampleRate = static_cast<Float64>(m_sample_rate);
	format.mFormatID = kAudioFormatLinearPCM;
	format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
	format.mChannelsPerFrame = m_output_channels;
	format.mBitsPerChannel = sizeof(SampleType) * 8;
	format.mBytesPerFrame = sizeof(SampleType) * m_output_channels;
	format.mFramesPerPacket = 1;
	format.mBytesPerPacket = format.mBytesPerFrame;
	status = AudioUnitSetProperty(m_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format,
		sizeof(format));
	if (status != noErr)
	{
		Error::SetStringFmt(error, "Failed to set CoreAudio stream format: {}", static_cast<s32>(status));
		Destroy();
		return false;
	}

	if (m_output_channels > 2)
	{
		// Describe our channel order so the unit can map it to whatever the device has.
		// Needs proper testing on real multichannel hardware.
		const size_t layout_size = offsetof(AudioChannelLayout, mChannelDescriptions) +
								   sizeof(AudioChannelDescription) * m_output_channels;
		std::unique_ptr<u8[]> layout_storage = std::make_unique<u8[]>(layout_size);
		std::memset(layout_storage.get(), 0, layout_size);
		AudioChannelLayout* layout = reinterpret_cast<AudioChannelLayout*>(layout_storage.get());
		layout->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
		layout->mNumberChannelDescriptions = m_output_channels;
		for (u32 i = 0; i < m_output_channels; i++)
			layout->mChannelDescriptions[i].mChannelLabel = setup.second[i];

		status = AudioUnitSetProperty(m_unit, kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Input, 0, layout,
			static_cast<UInt32>(layout_size));
		if (status != noErr)
			WARNING_LOG("Failed to set CoreAudio channel layout: {}", static_cast<s32>(status));
	}

	u32 latency_frames = GetBufferSizeForMS(
		m_sample_rate, (m_parameters.minimal_output_latency) ? m_parameters.buffer_ms : m_parameters.output_latency_ms);
	if (device != kAudioObjectUnknown)
	{
		// Buffer frame size is in device frames; the unit resamples if the device runs at a different rate.
		Float64 device_rate = 0.0;
		const AudioObjectPropertyAddress rate_addr = {
			kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
		UInt32 size = sizeof(device_rate);
		if (AudioObjectGetPropertyData(device, &rate_addr, 0, nullptr, &size, &device_rate) == noErr &&
			device_rate > 0.0 && static_cast<u32>(device_rate) != m_sample_rate)
		{
			latency_frames = static_cast<u32>(static_cast<Float64>(latency_frames) * device_rate / m_sample_rate);
		}

		const AudioObjectPropertyAddress addr = {
			kAudioDevicePropertyBufferFrameSizeRange, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
		AudioValueRange range = {};
		size = sizeof(range);
		if (AudioObjectGetPropertyData(device, &addr, 0, nullptr, &size, &range) == noErr)
		{
			const u32 min_frames = static_cast<u32>(range.mMinimum);
			const u32 max_frames = static_cast<u32>(range.mMaximum);
			DEV_LOG("CoreAudio buffer frame range: {} - {}", min_frames, max_frames);
			if (m_parameters.minimal_output_latency)
				latency_frames = min_frames;
			latency_frames = std::clamp(latency_frames, min_frames, std::max(min_frames, max_frames));
		}

		UInt32 frames = latency_frames;
		status = AudioUnitSetProperty(m_unit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &frames,
			sizeof(frames));
		if (status != noErr)
			WARNING_LOG("Failed to set CoreAudio buffer frame size to {}: {}", latency_frames, static_cast<s32>(status));
		else
			DEV_LOG("CoreAudio buffer frame size: {}", latency_frames);
	}

	// ReadFrames() can be asked for up to this many frames per callback.
	UInt32 max_frames_per_slice = 4096;
	AudioUnitSetProperty(m_unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
		&max_frames_per_slice, sizeof(max_frames_per_slice));

	AURenderCallbackStruct callback = {&CoreAudioAudioStream::RenderCallback, this};
	status = AudioUnitSetProperty(m_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback,
		sizeof(callback));
	if (status != noErr)
	{
		Error::SetStringFmt(error, "Failed to set CoreAudio render callback: {}", static_cast<s32>(status));
		Destroy();
		return false;
	}

	BaseInitialize(setup.first, stretch_enabled);

	status = AudioUnitInitialize(m_unit);
	if (status != noErr)
	{
		Error::SetStringFmt(error, "AudioUnitInitialize() failed: {}", static_cast<s32>(status));
		Destroy();
		return false;
	}
	m_initialized = true;

	status = AudioOutputUnitStart(m_unit);
	if (status != noErr)
	{
		Error::SetStringFmt(error, "AudioOutputUnitStart() failed: {}", static_cast<s32>(status));
		Destroy();
		return false;
	}

	return true;
}

OSStatus CoreAudioAudioStream::RenderCallback(void* ref_con, AudioUnitRenderActionFlags* flags,
	const AudioTimeStamp* timestamp, UInt32 bus, UInt32 num_frames, AudioBufferList* data)
{
	CoreAudioAudioStream* const this_ptr = static_cast<CoreAudioAudioStream*>(ref_con);

	// Interleaved format, so there's a single buffer.
	for (UInt32 i = 0; i < data->mNumberBuffers; i++)
	{
		AudioBuffer& buf = data->mBuffers[i];
		const u32 frames = std::min<u32>(num_frames, buf.mDataByteSize / (sizeof(SampleType) * this_ptr->m_output_channels));
		this_ptr->ReadFrames(static_cast<SampleType*>(buf.mData), frames);
	}

	return noErr;
}

void CoreAudioAudioStream::SetPaused(bool paused)
{
	if (paused == m_paused || !m_unit)
		return;

	const OSStatus status = paused ? AudioOutputUnitStop(m_unit) : AudioOutputUnitStart(m_unit);
	if (status != noErr)
	{
		ERROR_LOG("Could not {} CoreAudio stream: {}", paused ? "pause" : "resume", static_cast<s32>(status));
		return;
	}

	m_paused = paused;
}

std::unique_ptr<AudioStream> AudioStream::CreateCoreAudioStream(u32 sample_rate, const AudioStreamParameters& parameters,
	const char* device_name, bool stretch_enabled, Error* error)
{
	std::unique_ptr<CoreAudioAudioStream> stream = std::make_unique<CoreAudioAudioStream>(sample_rate, parameters);
	if (!stream->Initialize(device_name, stretch_enabled, error))
		stream.reset();
	return stream;
}

std::vector<AudioStream::DeviceInfo> AudioStream::GetCoreAudioOutputDevices()
{
	std::vector<AudioStream::DeviceInfo> ret;
	ret.emplace_back(std::string(), TRANSLATE_STR("AudioStream", "Default"), 0);

	for (const AudioObjectID dev : GetAllDevices())
	{
		if (!DeviceHasOutputChannels(dev))
			continue;

		std::string uid = GetDeviceStringProperty(dev, kAudioDevicePropertyDeviceUID);
		if (uid.empty())
			continue;

		std::string name = GetDeviceStringProperty(dev, kAudioObjectPropertyName);
		if (name.empty())
			name = uid;

		ret.emplace_back(std::move(uid), std::move(name), GetDeviceOutputLatencyFrames(dev));
	}

	return ret;
}
