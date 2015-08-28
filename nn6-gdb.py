import gdb
import itertools
import re

from gdb.FrameDecorator import FrameDecorator

class NnetFrameFilter():

    def __init__(self):
        # Frame filter attribute creation.
        #
        # 'name' is the name of the filter that GDB will display.
        #
        # 'priority' is the priority of the filter relative to other
        # filters.
        #
        # 'enabled' is a boolean that indicates whether this filter is
        # enabled and should be executed.

        self.name = "NnetFrameFilter"
        self.priority = 100
        self.enabled = True

        # Register this frame filter with the global frame_filters
        # dictionary.
        gdb.frame_filters[self.name] = self

    def filter(self, frame_iter):
        frame_iter = itertools.imap(NnetFrameDecorator,
                                    frame_iter)
        return frame_iter

class NnetFrameDecorator(FrameDecorator):

    def __init__(self, fobj):
        self.fobj = fobj
        super(NnetFrameDecorator, self).__init__(fobj)

    def function(self):
        frame = self.fobj.inferior_frame()
        name = self.process(str(frame.name()))

        if frame.type() == gdb.INLINE_FRAME:
            name = name + " [inlined]"

        return name

    def process(self, name):
        name = re.sub('nnet::vec_size<double, 1> *', 'V', name)
        name = re.sub('nnet::mat_size<double, 1> *', 'M', name)
        name = re.sub('boost::fusion::vector[0-9] *', 'A', name)
        name = re.sub('Eigen::Matrix<double, 1, -1, [^>]*>', 'Eigen::Vector', name)
        name = re.sub('Eigen::Matrix<double, -1, -1, [^>]*>', 'Eigen::Matrix', name)
        name = re.sub('Eigen::Map<Eigen::Vector, 1, Eigen::Stride<0, 0> >', 'Eigen::Map<Eigen::Vector>', name)
        name = re.sub('Eigen::Map<Eigen::Matrix, 1, Eigen::Stride<0, 0> >', 'Eigen::Map<Eigen::Matrix>', name)
        return name

NnetFrameFilter()
