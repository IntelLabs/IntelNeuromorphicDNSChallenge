import numpy as np
from snr import si_snr
from typing import Iterable
import soundfile as sf
from typing import Optional, Tuple

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.model import PyAsyncProcessModel


class AudioSource(AbstractProcess):
    """Audio source process. It invokes a new clean and noisy sample
    from the dataset iterable at a set interval and offset. At every
    hop_length, it sends out a chunk of audio data.

    Parameters
    ----------
    dataset : Iterable
        A dataset object instance that when indexed returns clean and noisy
        audio samples.
    hop_length : int
        Amount of audio samples to step at every timestep.
    interval : int
        The interval (in timesteps) between subsequent audio samples.
    offset : int, optional
        The offset/phase of sample read, by default 0.
    sample_idx : int, optional
        First sample index to begin reading from, by default 0.
    first_buffer : int, optional
        The hop in preloaded data, by default 0.
    """

    def __init__(self,
                 dataset: Iterable,
                 hop_length: int,
                 interval: int,
                 offset: int = 0,
                 sample_idx: int = 0,
                 first_buffer: int = 0) -> None:
        super().__init__(dataset=dataset,
                         hop_length=hop_length,
                         interval=interval,
                         offset=offset,
                         sample_idx=sample_idx)
        buffer = hop_length * interval
        self.hop_length = Var((1,), hop_length)
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        self.clean_data = Var(shape=(buffer,), init=np.zeros(buffer))
        self.noisy_data = Var(shape=(buffer,), init=np.zeros(buffer))
        self.clean_out = OutPort(shape=(hop_length,))
        self.noisy_out = OutPort(shape=(hop_length,))
        self.proc_params['saved_dataset'] = dataset
        self.proc_params['data_size'] = hop_length * interval
        self.proc_params['first_idx'] = first_buffer

    def preload_data(self, clean: np.ndarray, noisy: np.ndarray):
        """Preload data before the first dataset read.

        Parameters
        ----------
        clean : np.ndarray
            Clean audio sample.
        noisy : np.ndarray
            Noisy audio sample.
        """
        buffer_len = len(self.clean_data.init)

        if len(clean) < buffer_len:
            self.clean_data.init[:len(clean)] = clean
        else:
            self.clean_data.init = clean[:buffer_len]

        if len(noisy) < buffer_len:
            self.noisy_data.init[:len(noisy)] = noisy
        else:
            self.noisy_data.init = noisy[:buffer_len]


@implements(proc=AudioSource, protocol=LoihiProtocol)
@requires(CPU)
class PyAudioSourceModel(PyLoihiProcessModel):
    clean_data: np.ndarray = LavaPyType(np.ndarray, float)
    noisy_data: np.ndarray = LavaPyType(np.ndarray, float)
    hop_length: np.ndarray = LavaPyType(np.ndarray, int)
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)
    clean_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    noisy_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.data_idx = self.proc_params['first_idx']
        self.sample_idx = self.proc_params['sample_idx']
        self.data_size = self.proc_params['data_size']
        self.dataset = self.proc_params['saved_dataset']

    def run_spk(self) -> None:
        start = int(self.data_idx)
        stop = int(self.data_idx + self.hop_length)
        self.clean_out.send(self.clean_data[start: stop])
        self.noisy_out.send(self.noisy_data[start: stop])
        self.data_idx += self.hop_length
        if self.data_idx > self.data_size:
            self.data_idx = 0

    def post_guard(self) -> None:
        return (self.time_step - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        idx = self.sample_idx % len(self.dataset)
        noisy_data, clean_data, *_ = self.dataset[idx]

        if len(noisy_data) > self.data_size:
            print('WARNING: new data has different size. '
                  f'Expected {self.data_size}, found {len(self.noisy_data)}')
            self.data_size = len(self.noisy_data)
            self.noisy_data = np.zeros(len(noisy_data))
            self.clean_data = np.zeros(len(clean_data))

        self.noisy_data[:len(noisy_data)] = noisy_data
        self.clean_data[:len(clean_data)] = clean_data

        self.data_idx = 0
        self.sample_idx += 1
        if self.sample_idx == len(self.dataset):
            self.sample_idx = 0


class AudioReceiver(AbstractProcess):
    """Audio receiver process. It receives clean input and denoised input
    data one hop_length at a time and evaluates the SI-SNR of the collected
    audio once every interval.

    Parameters
    ----------
    hop_length : int
        Hop length of audio samples.
    num_samples : int
        Number of audio samples expected to evaluate SI-SNR.
    interval : int
        The interval (in timesteps) between subsequent audio samples.
    offset : int, optional
        The phase/offset of sample evaluation, by default 0.
    """

    def __init__(self,
                 hop_length: int,
                 num_samples: int,
                 interval: int,
                 offset: int = 0) -> None:
        super().__init__(hop_length=hop_length,
                         interval=interval,
                         offset=offset)
        buffer = hop_length * interval
        self.num_samples = num_samples
        self.hop_length = Var((1,), hop_length)
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        self.clean_data = Var(shape=(buffer,), init=np.zeros(buffer))
        self.denoised_data = Var(shape=(buffer,), init=np.zeros(buffer))
        self.si_snr = Var(shape=(num_samples,), init=np.zeros(num_samples))
        self.clean_inp = InPort(shape=(hop_length,))
        self.denoised_inp = InPort(shape=(hop_length,))
        self.proc_params['data_size'] = hop_length * interval


@implements(proc=AudioReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyAudioReceiverModel(PyLoihiProcessModel):
    clean_data: np.ndarray = LavaPyType(np.ndarray, float)
    denoised_data: np.ndarray = LavaPyType(np.ndarray, float)
    hop_length: np.ndarray = LavaPyType(np.ndarray, int)
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)
    si_snr: np.ndarray = LavaPyType(np.ndarray, float)
    clean_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    denoised_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.data_idx = 0
        self.sample_idx = -1
        self.data_size = self.proc_params['data_size']
        self.save_to = self.proc_params.get('save_to', None)

    def run_spk(self) -> None:
        start = int(self.data_idx)
        stop = int(self.data_idx + self.hop_length)
        self.clean_data[start: stop] = self.clean_inp.recv()
        self.denoised_data[start: stop] = self.denoised_inp.recv()
        self.data_idx += self.hop_length
        if self.data_idx > self.data_size:
            self.data_idx = 0

    def post_guard(self) -> None:
        return (self.time_step - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        self.si_snr[self.sample_idx] = si_snr(self.clean_data.astype(float),
                                              self.denoised_data.astype(float))
        if self.save_to is not None:
            sf.write(file=f'{self.save_to}_{self.sample_idx}.wav',
                     data=self.denoised_data,
                     samplerate=16000)
            print(
                f'Saved Audio sample to {self.save_to}_{self.sample_idx}.wav')
        self.clean_data = np.zeros_like(self.clean_data)
        self.denoised_data = np.zeros_like(self.denoised_data)
        self.data_idx = 0
        self.sample_idx += 1
        if self.sample_idx >= len(self.si_snr):
            self.sample_idx = 0


class STFT(AbstractProcess):
    """STFT process. Calculates FFT of n_fft audio data points in buffer.
    It advances the audio samples by hop_length every timestep. The output is
    the absolute and angle of the STFT complex value.

    Parameters
    ----------
    n_fft : int
        Size of audio data buffer.
    hop_length : int
        The shift of audio data points at every timestep.
    prefetch_data : np.ndarray, optional
        Intial buffer data to begin with. It is assumed to be zero if None. Ny default None.
    """

    def __init__(self, n_fft: int, hop_length: int,
                 prefetch_data: Optional[np.ndarray] = None) -> None:
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        fft_out = n_fft // 2 + 1
        self.audio_inp = InPort(shape=(hop_length,))
        self.abs_out = OutPort(shape=(fft_out,))
        self.arg_out = OutPort(shape=(fft_out,))
        self.proc_params['fft_out'] = fft_out
        self.proc_params['prefetch_data'] = prefetch_data


@implements(proc=STFT, protocol=LoihiProtocol)
@requires(CPU)
class PySTFTModel(PyLoihiProcessModel):
    audio_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    abs_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    arg_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.n_fft = self.proc_params['n_fft']
        self.fft_out = self.proc_params['fft_out']
        self.hop_length = self.proc_params['hop_length']
        self.audio_buffer = np.zeros(self.n_fft)
        if self.proc_params['prefetch_data'] is not None:
            prefetch_buffer = self.proc_params['prefetch_data'][-self.n_fft:]
            self.audio_buffer[-len(prefetch_buffer):] = prefetch_buffer

    def run_spk(self) -> None:
        hop_length = self.hop_length
        new_audio = self.audio_inp.recv()
        self.audio_buffer[:-hop_length] = self.audio_buffer[hop_length:]
        self.audio_buffer[-hop_length:] = new_audio
        spectrum = np.fft.fft(self.audio_buffer)
        self.abs_out.send(np.absolute(spectrum[:self.fft_out]))
        self.arg_out.send(np.angle(spectrum[:self.fft_out]))


class ISTFT(AbstractProcess):
    """ISTFT process. Calculates IFFT of n_fft audio spectrum points in buffer.
    It advances the audio samples by hop_length every timestep. The output is the reconstructed audio.

    Parameters
    ----------
    n_fft : int
        Size of audio data buffer.
    hop_length : int
        The shift of audio data points at every timestep.
    delay : int, optional
        Number of data buffer overlap to reconstruct the audio sample. It must be less than n_fft//hop_length. By default 0.
    """

    def __init__(self, n_fft: int, hop_length: int, delay: int = 0) -> None:
        super().__init__(n_fft=n_fft, hop_length=hop_length)
        fft_out = n_fft // 2 + 1
        self.abs_inp = InPort(shape=(fft_out,))
        self.arg_inp = InPort(shape=(fft_out,))
        self.audio_out = OutPort(shape=(hop_length,))
        self.proc_params['count'] = min(delay + 1, n_fft // hop_length)


@implements(proc=ISTFT, protocol=LoihiProtocol)
@requires(CPU)
class PyISTFTModel(PyLoihiProcessModel):
    abs_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    arg_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    audio_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.n_fft = self.proc_params['n_fft']
        self.hop_length = self.proc_params['hop_length']
        self.audio_buffer = np.zeros(self.n_fft)
        self.count = self.proc_params['count']

    def run_spk(self) -> None:
        hop_length = self.hop_length
        self.audio_buffer[:-hop_length] = self.audio_buffer[hop_length:]
        self.audio_buffer[-hop_length:] = 0
        stft_abs = self.abs_inp.recv()
        stft_arg = self.arg_inp.recv()
        stft = stft_abs * np.cos(stft_arg) + 1j * stft_abs * np.sin(stft_arg)
        spectrum = np.concatenate([stft, np.conjugate(stft[1:-1][::-1])])
        segment = np.fft.ifft(spectrum)
        self.audio_buffer += segment.real

        start_idx = (self.n_fft - self.count * hop_length)
        stop_idx = start_idx + hop_length
        self.audio_out.send(self.audio_buffer[start_idx:stop_idx] / self.count)


class FixedPtAmp(AbstractProcess):
    """Fixed point amplification process. It applies a gain and then
    quantizes the values.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the amplifier data.
    gain : float, optional
        The gain to be applied to the data before quantization, by default
        1<<6.
    """

    def __init__(self, shape: Tuple[int], gain: float = 1 << 6) -> None:
        super().__init__(shape=shape, gain=gain)
        self.gain = Var((1,), gain)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=FixedPtAmp, protocol=LoihiProtocol)
@requires(CPU)
class PyFixedPtAmpModel(PyLoihiProcessModel):
    gain: np.ndarray = LavaPyType(np.ndarray, float)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self) -> None:
        data = self.inp.recv()
        self.out.send(np.round(data * self.gain).astype(int))


class FloatPtAmp(AbstractProcess):
    """Floating point converter process. It applies a gain to properly
    interpret the fixed point input as a floating point number.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the amplifier data.
    gain : float, optional
        The gain of the amplifier, by default 1/(1 << 6).
    """

    def __init__(self, shape: Tuple[int], gain: float = 1 / (1 << 6)) -> None:
        super().__init__(shape=shape, gain=gain)
        self.gain = Var((1,), gain)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=FloatPtAmp, protocol=LoihiProtocol)
@requires(CPU)
class PyFloatPtAmpModel(PyLoihiProcessModel):
    gain: np.ndarray = LavaPyType(np.ndarray, float)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        raw_data = self.inp.recv()
        raw_data = (raw_data.astype(np.int32) << 8) >> 8
        self.out.send((raw_data * self.gain).astype(float))


class Bias(AbstractProcess):
    """Bias shift process. It adds a constant values to its input data.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the data.
    shift : float
        Bias value to be added.
    """

    def __init__(self, shape: Tuple[int], shift: float) -> None:
        super().__init__(shape=shape, shift=shift)
        self.shift = Var((1,), shift)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=Bias, protocol=LoihiProtocol)
@requires(CPU)
class PyBiasModel(PyLoihiProcessModel):
    shift: np.ndarray = LavaPyType(np.ndarray, float)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        self.out.send(self.inp.recv() + self.shift)


class AmplitudeMixer(AbstractProcess):
    """Process to mix amplitude of the STFT amplitude data. Applies mask to
    clean the STFT magnitue spectrum.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the data.
    """

    def __init__(self, shape: Tuple[int]) -> None:
        super().__init__(shape=shape)
        self.mask_inp = InPort(shape=shape)
        self.stft_inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=AmplitudeMixer, protocol=LoihiProtocol)
@requires(CPU)
class PyAmplitudeMixerModel(PyLoihiProcessModel):
    mask_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    stft_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        mask = self.mask_inp.recv() + 1
        mask = mask * (mask > 0)
        self.out.send(self.stft_inp.recv() * mask)


class DelayBuffer(AbstractProcess):
    """Delay process. Delays the input signal by a given amount.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the data.
    delay : int
        Delay amount.
    """

    def __init__(self, shape: Tuple[int], delay: int) -> None:
        super().__init__(shape=shape, delay=delay)
        self.buffer = Var((*shape, delay))
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=DelayBuffer, protocol=LoihiProtocol)
@requires(CPU)
class PyDelayBufferModel(PyLoihiProcessModel):
    buffer: np.ndarray = LavaPyType(np.ndarray, float)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        self.out.send(self.buffer[..., 0])
        self.buffer[..., :-1] = self.buffer[..., 1:]
        self.buffer[..., -1] = self.inp.recv()


class Encoder(AbstractProcess):
    """Audio encoder process. Encapsulates STFT conversion, bias shift
    and fixed point conversion of the audio sample preprocessing.

    Parameters
    ----------
    n_fft : int
        Number of fft samples.
    hop_length : int
        Advance of audio samples at each timestep.
    bias : float
        Bias shift value.
    gain : float
        Fixed point conversion gain.
    prefetch_data : np.ndarray, optional
        Pre-fetched STFT data buffer. None will initialize the buffer to
        zero, by default None.
    """

    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 bias: float,
                 gain: float,
                 prefetch_data: np.ndarray = None) -> None:
        super().__init__(n_fft=n_fft, hop_length=hop_length,
                         bias=bias, gain=gain)
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                         prefetch_data=prefetch_data)
        self.bias = Bias(shape=self.stft.abs_out.shape, shift=bias)
        self.amplifier = FixedPtAmp(self.stft.abs_out.shape, gain=1 << 6)

        self.audio_inp = InPort(shape=self.stft.audio_inp.shape)
        self.abs_out = OutPort(shape=self.stft.abs_out.shape)
        self.arg_out = OutPort(shape=self.stft.abs_out.shape)
        self.enc_out = OutPort(shape=self.stft.abs_out.shape)

        self.audio_inp.connect(self.stft.audio_inp)
        self.stft.abs_out.connect(self.bias.inp)
        self.bias.out.connect(self.amplifier.inp)
        self.stft.abs_out.connect(self.abs_out)
        self.stft.arg_out.connect(self.arg_out)
        self.amplifier.out.connect(self.enc_out)


@implements(proc=Encoder, protocol=LoihiProtocol)
class PyEncoderModel(AbstractSubProcessModel):
    def __init__(self, proc: AbstractProcess) -> None:
        self.audio_inp: PyInPort = LavaPyType(np.ndarray, float)
        self.abs_out: PyOutPort = LavaPyType(np.ndarray, float)
        self.abs_out: PyOutPort = LavaPyType(np.ndarray, float)
        self.enc_out: PyOutPort = LavaPyType(np.ndarray, int)


class Decoder(AbstractProcess):
    """Decoder process of network output. Reconstructs the audio data from
    the raw denoising network output.

    Parameters
    ----------
    n_fft : int
        Number of fft samples.
    hop_length : int
        Advance of audio samples at each timestep.
    gain : float
        Floating point interpretation gain.
    istft_delay : int
        Delay of ISTFT audio generation. It must be less than n_fft//hop_length.
    """

    def __init__(self, n_fft: int, hop_length: int, gain: float,
                 istft_delay: int) -> None:
        super().__init__(n_fft=n_fft, hop_length=hop_length, gain=gain)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length,
                           delay=istft_delay)  # delay can be [0, n_fft/hop_len)
        self.amplifier = FloatPtAmp(shape=self.istft.abs_inp.shape, gain=gain)
        self.mixer = AmplitudeMixer(shape=self.istft.abs_inp.shape)

        self.dec_inp = InPort(shape=self.istft.abs_inp.shape)
        self.abs_inp = InPort(shape=self.istft.abs_inp.shape)
        self.arg_inp = InPort(shape=self.istft.abs_inp.shape)
        self.audio_out = OutPort(shape=self.istft.audio_out.shape)

        self.dec_inp.connect(self.amplifier.inp)
        self.amplifier.out.connect(self.mixer.mask_inp)
        self.abs_inp.connect(self.mixer.stft_inp)
        self.mixer.out.connect(self.istft.abs_inp)
        self.arg_inp.connect(self.istft.arg_inp)
        self.istft.audio_out.connect(self.audio_out)


@implements(proc=Decoder, protocol=LoihiProtocol)
class PyDecoderModel(AbstractSubProcessModel):
    def __init__(self, proc: AbstractProcess) -> None:
        self.dec_inp: PyInPort = LavaPyType(np.ndarray, int)
        self.abs_inp: PyInPort = LavaPyType(np.ndarray, float)
        self.arg_inp: PyInPort = LavaPyType(np.ndarray, float)
        self.audio_out: PyOutPort = LavaPyType(np.ndarray, float)


proc_model_map = {AudioSource: PyAudioSourceModel,
                  AudioReceiver: PyAudioReceiverModel,
                  STFT: PySTFTModel,
                  ISTFT: PyISTFTModel,
                  FloatPtAmp: PyFloatPtAmpModel,
                  FixedPtAmp: PyFixedPtAmpModel,
                  AmplitudeMixer: PyAmplitudeMixerModel,
                  Bias: PyBiasModel,
                  DelayBuffer: PyDelayBufferModel}


def get_var_dict(model_class) -> dict:
    var_names = [v for v, m in vars(model_class).items()
                 if not (v.startswith('_') or callable(m))]
    var_dict = {var_name: getattr(model_class, var_name)
                for var_name in var_names}
    if model_class == PyLoihiProcessModel:
        return {}
    for base in model_class.__bases__:
        var_dict = {**get_var_dict(base), **var_dict}
    return var_dict


def get_callable_dict(model_class) -> dict:
    callable_names = [v for v, m in vars(model_class).items()
                      if callable(m) and not v.startswith('_')]
    callable_dict = {callable_name: getattr(model_class, callable_name)
                     for callable_name in callable_names}
    if model_class == PyLoihiProcessModel:
        return {}
    for base in model_class.__bases__:
        callable_dict = {**get_callable_dict(base), **callable_dict}
    return callable_dict


def PyLoihiModelToPyAsyncModel(py_loihi_model: PyLoihiProcessModel):
    # based on the constructor of PyLoihiProcessModel and PyAsyncProcModel
    exclude_vars = ['time_step', 'phase']
    exclude_callables = ['run_spk',
                         'pre_guard', 'run_pre_mgmt',
                         'post_guard', 'run_post_mgmt',
                         'implements_process', 'implements_protocol']
    name = py_loihi_model.__name__ + 'Async'
    var_dict = get_var_dict(py_loihi_model)
    var_dict['implements_process'] = py_loihi_model.implements_process
    var_dict['implements_protocol'] = AsyncProtocol
    callable_dict = {k: v for k, v in get_callable_dict(py_loihi_model).items()
                     if k not in exclude_callables}

    def __init__(self, proc_params: dict):
        PyAsyncProcessModel.__init__(self, proc_params)
        ref_model = py_loihi_model(proc_params)
        attributes = [v for v, m in vars(ref_model).items()
                      if not (v.startswith('_') or callable(m))
                      and v not in var_dict.keys()
                      and v not in vars(self)
                      and v not in exclude_vars]
        for attr in attributes:
            setattr(self, attr, getattr(ref_model, attr))
        self.time_step = 1

    def run_async(self) -> None:
        while self.time_step != self.num_steps + 1:
            if py_loihi_model.pre_guard(self):
                py_loihi_model.run_pre_mgmt(self)
            py_loihi_model.run_spk(self)
            if py_loihi_model.post_guard(self):
                py_loihi_model.run_post_mgmt(self)
            self.time_step += 1

    newclass = type(name, (PyAsyncProcessModel,),
                    {'__init__': __init__,
                     'run_async': run_async,
                     **var_dict,
                     **callable_dict})
    return newclass


async_proc_model_map = {key: PyLoihiModelToPyAsyncModel(val)
                        for (key, val) in proc_model_map.items()}
