import matplotlib.pyplot as plt
import numpy as np
import h5py

plt.rcParams['font.family'] = ['euclid']

def stft(data, fs, fc_CH, start_time, end_time):
    """
    :param data: data
    :param fs: sampling rate
    :param fc_CH: channel 2440/5800
    :param start_time
    :param end_time
    :return:
    """
    if fc_CH == 2440:
        i = data['RF0_I'][0]
        q = data['RF0_Q'][0]
    else:
        i = data['RF1_I'][0]
        q = data['RF1_Q'][0]
    iq = i + 1j * q
    iq = iq[int(start_time*fs):int(end_time * fs)]
    length = iq.shape[0]
    freq_bins = 2048
    adding = int(0.5 * freq_bins)
    time_bins = int((iq.shape[0] - freq_bins) / adding)
    map = np.zeros((time_bins, freq_bins))
    for i in range(time_bins):
        iq_seg = iq[i * adding: (i * adding + freq_bins)]
        x_fft = np.fft.fft(iq_seg * np.hamming(freq_bins)) / (iq_seg.shape[0] - 1)
        x_fft = np.fft.fftshift(x_fft)
        map[i] = 10 * np.log10(abs(x_fft))
    del i
    del q
    del iq
    t = np.arange(0, time_bins, 1) / time_bins * (length / fs)
    f = (np.arange(0, freq_bins, 1) - freq_bins / 2) / freq_bins * fs / 1e+6 + fc_CH
    plt.figure(figsize=(6, 4), dpi=150)
    cmap = plt.cm.jet
    plt.pcolormesh(t, f, np.transpose(map), shading='auto', cmap=cmap)
    cb = plt.colorbar()
    plt.clim(-80, -20)
    plt.title('STFT Analysis on ISM' + ' ' + str(round(fc_CH/1000, 1)) + ' ' + 'GHz')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cb.ax.tick_params(labelsize=12)
    plt.xlim([0.001, end_time-start_time - 0.001])
    cb.set_label('Intensity(dB)', fontname='STSONG', fontsize=12)
    plt.xlabel('Time(s)', fontname='STSONG', fontsize=12)
    plt.ylabel('Freq(MHz)', fontname='STSONG', fontsize=12)
    plt.tight_layout()
    del map
    plt.show()


if __name__ == "__main__":
    fs = 100e+6
    data = h5py.File('.\\T0101_D10_S0000.mat', 'r')
    stft(data, fs, fc_CH=2440, start_time=0.1, end_time=0.2)


