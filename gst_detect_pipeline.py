import cv2
import gi
import numpy as np
import pdb 
import multiprocessing as mp 

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst,GObject,GLib
import time
import sys
GObject.threads_init()
Gst.init(None)
from gst_lt_infer import test

#Gst.debug_set_active(True)
#Gst.debug_set_default_threshold(4)

#src_address = '/workspace/demo.mp4'
#input_type = 'localfile' # 'rtsp'
src_address = 'https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm'
src_address = 'http://112.74.200.9:88/tv000000/m3u8.php?/migu/624878396'
src_address = 'http://cbsnewshd-lh.akamaihd.net/i/CBSNHD_7@199302/master.m3u8'
count = 0

class Video():
    

    def __init__(self, frame_queue, state):
       
        self._frame = None
        self.frame_queue = frame_queue
        self.state = state
        self.cmd = 'uridecodebin uri={} ! videoconvert ! videoscale ! appsink name=sink caps="video/x-raw,format=BGR,pixel-aspect-ratio=1/1" '.format(src_address)
        #cmd = 'uridecodebin uri={} ! videoconvert ! videoscale ! autovideosink '.format(src_address)
        print(self.cmd)
        self.run()

    #def frame(self):
    #    return self._frame
    #def frame_available(self):
    #    print('check frame')
    #    return type(self._frame) != type(None)
    
    @staticmethod
    def buffer_to_opencv(pad, buf):
        if not buf:
          return Gst.PadProbeReturn.OK
        caps = pad.get_current_caps()
        h,w = caps.get_structure(0).get_value('height'),caps.get_structure(0).get_value('width')
        #is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        #if is_mapped:
        #  try:
        #    arr = np.ndarray((h,w,3), buffer=map_info.data, dtype=np.uint8).copy()
        #    print(arr[100])
        #    #print('h,w',h,w)
        #  finally:
        #    buf.unmap(map_info)
        # another way of get buffer data
        arr = np.ndarray((h,w,3), buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return arr
    def probe_callback(self, pad, info):
        buf = info.get_buffer()
        if not buf:
          return Gst.PadProbeReturn.OK
        arr = self.buffer_to_opencv(pad,buf)
        if not self.frame_queue.full():
          self.frame_queue.put(arr, block=False)
        else:
          #print('queue is full, skipping this frame')
          pass
        #print('set frame' )
        #cv2.imshow('frame',self._frame)
        if self.state.is_set():
          #print('check loop quit')
          struct = Gst.Structure('Stop')
          struct.set_value('stop',True)
          self.bus.post(Gst.Message.new_application(self.bus,struct))
        
        return Gst.PadProbeReturn.OK

    @staticmethod
    def bus_call(bus,message,loop):
        t = message.type
        if t == Gst.MessageType.EOS:
          sys.stdout.write('End of the stream\n')
          loop.quit()
        elif t == Gst.MessageType.ERROR:
          err, debug = message.parse_error()
          sys.stderr.write("Error: %s: %s\n" % (err, debug))
          loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
          # if message_src = pipe
          pass
          #print('state change from xx to xx')
        elif t == Gst.MessageType.APPLICATION:
          print('received a application msg')
          if message.has_name('Stop'):
            print('receive stop msg, stop')
            loop.quit()
        return True

    def sample_to_opencv(self,sample):
        buf = sample.get_buffer()
        caps = samples.get_caps()
        h,w = caps.get_structure(0).get_value('height'),caps.get_structure(0).get_value('width')
        arr = np.ndarray((h,w,3), buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        print('hw ',h,w)
        return arr

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        arr = self.sample_to_opencv(sample)
        #self._frame = arr
        try:
          self.frame_queue.put(arr,block=false)
        except queue.Full:
          print('frame_queue is full')

        print('get frame')
        return Gst.FlowReturn.OK

    def run(self):
        loop = GObject.MainLoop()
        self.loop = loop
        pipe = Gst.parse_launch(self.cmd)
        sink = pipe.get_by_name('sink')
        
        pipe.set_state(Gst.State.PLAYING)
        sink.connect('new-sample', self.new_buffer, sink)
        print('connect sample')
        # TBD connect add-paded signal to  uridecode
        srcpad = sink.get_static_pad('sink')
        probeID = srcpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_callback)

        bus = pipe.get_bus()
        self.bus = bus
        bus.add_signal_watch()
        bus.connect('message', self.bus_call, loop)
        try:
          loop.run()
        except:
          pass
        # cleanup
        pipe.set_state(Gst.State.NULL)
        while not self.frame_queue.empty():
          try:
            _ = self.frame_queue.get()
            print('pop queue')
          except:
            break
        print('stopping')
        

def gst_proc(frame_queue, state):
  video = Video(frame_queue, state)


def main():
  frame_queue = mp.Queue(maxsize=1024)
  state = mp.Event()
  gst_process = mp.Process(target=gst_proc, args=(frame_queue,state))
  
  gst_process.start()

  count =0
  try:
    while True:
      if not frame_queue.empty():
        frame = frame_queue.get()
        #img = cv2.resize(frame, tuple([224, 224]))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        post_frame = test(frame, frame_id=count )
        #img = cv2.resize(img_ori, tuple([input_h_w, input_h_w]))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', post_frame)
        count += 1
      #print('play video')
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('break play')
  except KeyboardInterrupt:
    print('keyboard interrupt')
  except:
    e = sys.exc_info()
    print('caught main exception')
    print(e)

  if state is not None:
    state.set()
    print('set state to stop')
    
    gst_process.terminate()
    gst_process.join()
 
    print('close queue')

  
  cv2.destroyAllWindows()


if __name__ == '__main__': 
    #sys.exit(main())
    main()
    
    
  

    
