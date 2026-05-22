FROM python:3.11-slim

# Reproducible dependency set for the NAT harness. Regenerate with:
#   uv pip compile requirements.in -o requirements.lock
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

COPY run.py /opt/nsl/run.py

ARG NSL_BUILD_CONTEXT_SHA=unknown
LABEL nsl.build-context-sha=$NSL_BUILD_CONTEXT_SHA

# NAT/logging dependencies may write relative to CWD when imported; /tmp is
# writable for the non-root uid used by ContainerHarness.
WORKDIR /tmp

ENTRYPOINT []
CMD ["python", "/opt/nsl/run.py"]
