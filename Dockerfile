# Use Alpine Linux for minimal size
FROM python:3.11-alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install essential system dependencies and compilers
RUN apk add --no-cache \
    # Compilers and build tools
    gcc \
    g++ \
    musl-dev \
    make \
    cmake \
    pkgconfig \
    # Fortran compiler
    gfortran \
    # Additional build tools
    autoconf \
    automake \
    libtool \
    # System libraries
    libc6-compat \
    # Graphics and OpenGL libraries
    mesa-gl \
    mesa-gles \
    # BLAS and LAPACK
    openblas \
    openblas-dev \
    # Image processing libraries
    libjpeg-turbo-dev \
    libpng-dev \
    tiff-dev \
    # Additional system libraries
    zlib-dev \
    bzip2-dev \
    # OpenCV dependencies (correct Alpine names)
    ffmpeg-dev \
    # Other essential libraries
    curl \
    # Development headers
    linux-headers \
    # Python development
    python3-dev \
    py3-pip

# Set environment variables for BLAS and OpenCV
ENV BLAS=/usr/lib/libopenblas.so
ENV OPENBLAS_NUM_THREADS=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=2147483647

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with proper build flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include" --global-option="-L/usr/lib" -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser -D app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the real working application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

