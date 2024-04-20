FROM nvcr.io/nvidia/pytorch:23.11-py3
WORKDIR /workspace
ADD ./LLM-common-eval/requirements.txt r1.txt
RUN pip install -r r1.txt
ADD ./medusa/requirements.txt r2.txt
RUN pip install -r r2.txt
ADD . s3d
WORKDIR /workspace/s3d
RUN cd medusa && pip install -e .
CMD /bin/bash
