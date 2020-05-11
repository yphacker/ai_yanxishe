# coding=utf-8
# author=yphacker


import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


class AudioProcessor(object):

    def prepare_processing_graph(self, model_settings):
        """Builds a TensorFlow graph to apply the input distortions.

        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.

        Args:
          model_settings: Information about the current model being trained.
        """
        desired_samples = model_settings['desired_samples']
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                     self.time_shift_offset_placeholder_,
                                     [desired_samples, -1])
        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                           [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
        background_mul = tf.multiply(self.background_data_placeholder_,
                                     self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)
        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['dct_coefficient_count'])

    def get_data(self, filename, model_settings, time_shift, sess):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          model_settings: Information about the current model being trained.
          background_frequency: How many clips will have background noise, 0.0 to
            1.0.
          background_volume_range: How loud the background noise will be.
          time_shift: How much to randomly shift the clips by in time.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of labels in
          one-hot form.
        """
        sample_count = 1
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        desired_samples = model_settings['desired_samples']
        # If we're time shifting, set up the offset for this sample.
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]
        input_dict = {
            self.wav_filename_placeholder_: filename,
            self.time_shift_padding_placeholder_: time_shift_padding,
            self.time_shift_offset_placeholder_: time_shift_offset,
        }
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
        input_dict[self.background_data_placeholder_] = background_reshaped
        input_dict[self.background_volume_placeholder_] = background_volume
        input_dict[self.foreground_volume_placeholder_] = 1
        # Run the graph to produce the output audio.
        tmp = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
        data[0, :] = tmp
        return data


    def get_unprocessed_data(self, filename, model_settings):
        """Retrieve sample data for the given partition, with no transformations.

        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          model_settings: Information about the current model being trained.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.

        Returns:
          List of sample data for the samples, and list of labels in one-hot form.
        """
        desired_samples = model_settings['desired_samples']
        data = np.zeros((1, desired_samples))
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(
                wav_loader, desired_channels=1, desired_samples=desired_samples)
            foreground_volume_placeholder = tf.placeholder(tf.float32, [])
            scaled_foreground = tf.multiply(wav_decoder.audio,
                                            foreground_volume_placeholder)
            input_dict = {wav_filename_placeholder: filename}
            input_dict[foreground_volume_placeholder] = 1
            data[0, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        return data
