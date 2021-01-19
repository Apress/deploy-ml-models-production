FROM python:3.5-slim
WORKDIR /app
ADD . /app

RUN pip install flask
RUN pip install numpy>=1.9.2

RUN pip install scikit-learn>=0.18

RUN pip install pandas>=0.19


EXPOSE 5000

ENV NAME Opentoall

CMD ["python","app.py"]


