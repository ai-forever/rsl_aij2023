FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda11.1-pt1.9.1:0.0.32.3
USER root
RUN python -m pip install openmim==0.3.9
RUN python -m mim install mmengine
RUN python -m mim install mmcv
RUN python -m mim install mmaction2
USER user
WORKDIR /home/jovyan/