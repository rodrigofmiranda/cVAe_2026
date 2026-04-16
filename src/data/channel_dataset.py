#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: channel_dataset_full_square
# GNU Radio version: 3.10.7.0

from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time


def _build_interleaved_iq_int16(samples, source_mode="square", seed=None, circle_radius_scale=1.0):
    """Build interleaved int16 IQ samples for the selected support geometry."""
    n = int(samples)
    if n <= 0:
        raise ValueError("samples must be > 0")

    mode = str(source_mode).strip().lower()
    rng = numpy.random.default_rng(seed)

    if mode in ("square", "full_square"):
        i = rng.integers(-32767, 32767, size=n, dtype=numpy.int16)
        q = rng.integers(-32767, 32767, size=n, dtype=numpy.int16)
    elif mode in ("circle", "circular", "full_circle"):
        # Uniform-in-area disk: r = R*sqrt(U), theta = 2*pi*V
        # circle_radius_scale allows controlled power sweeps while preserving
        # a circular support shape. Values >1 can clip at int16 bounds.
        theta = rng.uniform(0.0, 2.0 * numpy.pi, size=n)
        radius = float(circle_radius_scale) * 32767.0 * numpy.sqrt(rng.uniform(0.0, 1.0, size=n))
        i = numpy.rint(radius * numpy.cos(theta)).astype(numpy.int32)
        q = numpy.rint(radius * numpy.sin(theta)).astype(numpy.int32)
        i = numpy.clip(i, -32767, 32767).astype(numpy.int16)
        q = numpy.clip(q, -32767, 32767).astype(numpy.int16)
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode!r}")

    interleaved = numpy.empty(2 * n, dtype=numpy.int16)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved




class channel_dataset(gr.top_block):

    def __init__(self, source_mode="square", seed=None, sent_fname=None, recv_fname=None, circle_radius_scale=1.0):
        source_mode_norm = str(source_mode).strip().lower()
        if source_mode_norm in ("circle", "circular", "full_circle"):
            flowgraph_name = "channel_dataset_full_circle"
            default_dir = "/home/rodrigo/Documents/Amplificado/FULLCIRCLE"
        else:
            flowgraph_name = "channel_dataset_full_square"
            default_dir = "/home/rodrigo/Documents/Amplificado/FULLSQUARE"

        gr.top_block.__init__(self, flowgraph_name, catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 4
        self.nfilts = nfilts = 45
        self.tuning = tuning = 1000e3
        self.source_mode = source_mode = source_mode_norm
        self.seed = seed
        self.circle_radius_scale = circle_radius_scale
        self.sent_fname = sent_fname = sent_fname or f"{default_dir}/sent.bin"
        self.samples = samples = 2000e3
        self.samp_rate = samp_rate = 200e3
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), 0.35, 45*nfilts)
        self.rf_gain = rf_gain = 1
        self.recv_fname = recv_fname = recv_fname or f"{default_dir}/received.bin"
        self.phase_bw = phase_bw = 0.00628
        self.excess_bw = excess_bw = 0.35
        self.arity = arity = 4

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("addr0=192.168.10.3", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('external', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(tuning, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_bandwidth((samp_rate/2), 0)
        self.uhd_usrp_source_0.set_normalized_gain(rf_gain, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("addr0=192.168.10.2", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_clock_source('external', 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(tuning, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_bandwidth(samp_rate, 0)
        self.uhd_usrp_sink_0.set_normalized_gain(rf_gain, 0)
        self.root_raised_cosine_filter_0 = filter.interp_fir_filter_ccf(
            sps,
            firdes.root_raised_cosine(
                0.6,
                samp_rate,
                (samp_rate/sps),
                0.35,
                nfilts))
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(sps, phase_bw, rrc_taps, nfilts, (nfilts/2), 1.5, 1)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_cc(1000)
        self.blocks_interleaved_short_to_complex_0 = blocks.interleaved_short_to_complex(False, False,32767)
        self.blocks_head_0_0_0 = blocks.head(gr.sizeof_gr_complex*1, int(samples))
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1, recv_fname, False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, sent_fname, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_random_source_x_1 = blocks.vector_source_s(
            list(
                map(
                    int,
                    _build_interleaved_iq_int16(
                        samples=samples,
                        source_mode=source_mode,
                        seed=seed,
                        circle_radius_scale=circle_radius_scale,
                    ),
                )
            ),
            False,
        )


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_1, 0), (self.blocks_interleaved_short_to_complex_0, 0))
        self.connect((self.blocks_head_0_0_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_interleaved_short_to_complex_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.blocks_head_0_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_multiply_const_vxx_1, 0))


    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 45*self.nfilts))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(0.6, self.samp_rate, (self.samp_rate/self.sps), 0.35, self.nfilts))

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), 0.35, 45*self.nfilts))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(0.6, self.samp_rate, (self.samp_rate/self.sps), 0.35, self.nfilts))

    def get_tuning(self):
        return self.tuning

    def set_tuning(self, tuning):
        self.tuning = tuning
        self.uhd_usrp_sink_0.set_center_freq(self.tuning, 0)
        self.uhd_usrp_sink_0.set_center_freq(self.tuning, 1)
        self.uhd_usrp_source_0.set_center_freq(self.tuning, 0)

    def get_sent_fname(self):
        return self.sent_fname

    def set_sent_fname(self, sent_fname):
        self.sent_fname = sent_fname
        self.blocks_file_sink_0.open(self.sent_fname)

    def get_samples(self):
        return self.samples

    def set_samples(self, samples):
        self.samples = samples
        self.blocks_head_0_0_0.set_length(int(self.samples))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(0.6, self.samp_rate, (self.samp_rate/self.sps), 0.35, self.nfilts))
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_bandwidth(self.samp_rate, 0)
        self.uhd_usrp_sink_0.set_bandwidth(self.samp_rate, 1)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_bandwidth((self.samp_rate/2), 0)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.digital_pfb_clock_sync_xxx_0.update_taps(self.rrc_taps)

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self.uhd_usrp_sink_0.set_normalized_gain(self.rf_gain, 0)
        self.uhd_usrp_sink_0.set_gain(self.rf_gain, 1)
        self.uhd_usrp_source_0.set_normalized_gain(self.rf_gain, 0)

    def get_recv_fname(self):
        return self.recv_fname

    def set_recv_fname(self, recv_fname):
        self.recv_fname = recv_fname
        self.blocks_file_sink_0_0.open(self.recv_fname)

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_pfb_clock_sync_xxx_0.set_loop_bandwidth(self.phase_bw)

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw

    def get_arity(self):
        return self.arity

    def set_arity(self, arity):
        self.arity = arity




def main(top_block_cls=channel_dataset, options=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--source_mode",
        default="square",
        choices=["square", "circle"],
        help="IQ support geometry for transmitted source samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible source generation.",
    )
    parser.add_argument(
        "--sent_fname",
        default=None,
        help="Optional output path for TX complex stream.",
    )
    parser.add_argument(
        "--recv_fname",
        default=None,
        help="Optional output path for RX complex stream.",
    )
    parser.add_argument(
        "--circle_radius_scale",
        type=float,
        default=1.0,
        help="Scale factor for circle radius before int16 clipping.",
    )
    args = parser.parse_args()

    tb = top_block_cls(
        source_mode=args.source_mode,
        seed=args.seed,
        sent_fname=args.sent_fname,
        recv_fname=args.recv_fname,
        circle_radius_scale=args.circle_radius_scale,
    )

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
