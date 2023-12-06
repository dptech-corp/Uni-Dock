FROM dpzhengh/mcdock:base

COPY . /opt/mcdock
RUN cd /opt/mcdock && \
    python setup.py install