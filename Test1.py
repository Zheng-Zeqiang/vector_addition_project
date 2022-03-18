#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pynq
from pynq import Overlay

overlay = Overlay("/home/xilinx/jupyter_notebooks/vector_add_overlay/design1/design_1v.bit")
print(overlay)
#overlay.ip_dict
vecadd_inst = overlay.vector_add_0
print("Bitstream loaded")


# In[5]:


array_a = pynq.allocate(shape=(1024,), dtype=np.int32)
array_b = pynq.allocate(shape=(1024,), dtype=np.int32)
array_c = pynq.allocate(shape=(1024,), dtype=np.int32)


for i in range(1024):
    array_a[i] = float(i)
    array_b[i] = float(i)
    array_c[i] = 0.0


array_a.sync_to_device()
array_b.sync_to_device()
array_c.sync_to_device()


print('finish preparing arrays, starting accelerator...')


# In[6]:


vecadd_inst.call(array_a, array_b, array_c)
handle = vecadd_inst.start(array_a, array_b, array_c)
handle.wait()

print('accelerator done')


# In[ ]:


array_a.sync_from_device()
array_b.sync_from_device()
array_c.sync_from_device()

for i in range(10):
    print(f'array_a[{i}] = {array_a[i]}')
for i in range(10):
    print(f'array_b[{i}] = {array_b[i]}')
for i in range(10):
    print(f'array_c[{i}] = {array_c[i]}')

array_a.close()
array_b.close()
array_c.close()

