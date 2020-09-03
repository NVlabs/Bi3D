FROM nvcr.io/nvidia/pytorch:19.01-py3

RUN pip install Pillow
RUN pip install imageio
RUN pip install tensorboardX
RUN pip install opencv-python
