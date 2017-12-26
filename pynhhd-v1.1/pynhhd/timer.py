'''
Copyright (c) 2015, Harsh Bhatia (bhatia4@llnl.gov)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy
import timeit

class Timer(object):

    def display(self):

        tseconds = self.etime - self.stime

        print ' [[ elapsed time:',
        if tseconds < numpy.power(10.,-6.):
            print("%.3f micro-sec." % (tseconds*numpy.power(10,6))),

        elif tseconds < numpy.power(10.,-3.):
            print("%.3f milli-sec." % (tseconds*numpy.power(10.,3.))),

        elif tseconds < 60.0:
            print("%.3f sec." % (tseconds)),

        else:
            m = int(tseconds/ 60.0)
            print("%d min. %.3f sec." % (m, (tseconds - 60*m))),

        print ']]'

    def start(self):
        self.stime = timeit.default_timer()

    def end(self, print_time=True):
        self.etime = timeit.default_timer()

        if print_time:
            self.display()

    def __init__(self):
        self.start()
