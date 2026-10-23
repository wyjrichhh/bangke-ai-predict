# Build CTranslate2 as a static library for librime-predict.
#
# Static linking ensures the plugin (.dylib/.so) is self-contained:
# users don't need to install CTranslate2 separately.
#
# Usage:
#   make -f deps.mk              # build with default settings
#   make -f deps.mk clean        # remove build artifacts

project_root = $(CURDIR)
deps_dir = $(project_root)/deps

ifndef NOPARALLEL
NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
export MAKEFLAGS += -j$(NPROC)
endif

build ?= build
prefix ?= $(project_root)

OS_NAME = $(shell uname)

predict_deps = ctranslate2

.PHONY: all clean clean-dist $(predict_deps)

all: $(predict_deps)

clean:
	rm -rf $(deps_dir)/CTranslate2/$(build) || true

clean-dist:
	rm -f $(prefix)/lib/libctranslate2* || true
	rm -f $(prefix)/lib/libcpu_features* || true
	rm -rf $(prefix)/include/ctranslate2 || true

ctranslate2:
	cd "$(deps_dir)/CTranslate2" && \
	cmake . -B$(build) \
	-DBUILD_SHARED_LIBS:BOOL=OFF \
	-DBUILD_CLI:BOOL=OFF \
	-DBUILD_TESTS:BOOL=OFF \
	-DWITH_CUDA:BOOL=OFF \
	-DWITH_MKL:BOOL=OFF \
	-DWITH_OPENBLAS:BOOL=OFF \
	-DWITH_RUY:BOOL=ON \
	-DWITH_ACCELERATE:BOOL=$(if $(filter Darwin,$(OS_NAME)),ON,OFF) \
	-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.10 \
	-DOPENMP_RUNTIME:STRING="NONE" \
	-DCMAKE_BUILD_TYPE:STRING="Release" \
	-DCMAKE_INSTALL_PREFIX:PATH="$(prefix)" \
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
	&& cmake --build $(build) \
	&& cmake --build $(build) --target install
	@# cpu_features is built only on x86_64; copy it to prefix if present.
	@if [ -f "$(deps_dir)/CTranslate2/$(build)/third_party/cpu_features/libcpu_features.a" ]; then \
		cp "$(deps_dir)/CTranslate2/$(build)/third_party/cpu_features/libcpu_features.a" "$(prefix)/lib/"; \
		echo "Installed cpu_features.a to $(prefix)/lib/"; \
	fi
	@# RUY's static dependencies must be included in the archive used by the plugin.
	@# MRI uses simple aliases because its parser does not support quoted paths.
	set -eu; \
	ruy_dir=$$(cd "$(deps_dir)/CTranslate2/$(build)/third_party/ruy" && pwd); \
	set -- "$$ruy_dir/ruy"/libruy*.a \
		"$$ruy_dir/ruy/profiler/libruy_profiler_instrumentation.a" \
		"$$ruy_dir/third_party/cpuinfo/libcpuinfo.a" \
		"$$ruy_dir/third_party/cpuinfo/deps/clog/libclog.a"; \
	for archive do test -f "$$archive"; $(AR) -t "$$archive" > /dev/null; done; \
	case "$(OS_NAME)" in \
		Darwin) libtool -static -o "$(prefix)/lib/libruy.a" "$$@" ;; \
		Linux) \
			staging=$$(mktemp -d); \
			trap 'rm -rf "$$staging"' 0; \
			printf 'CREATE merged.a\n' > "$$staging/merge.mri"; \
			index=0; \
			for archive do \
				ln -s "$$archive" "$$staging/input-$$index.a"; \
				printf 'ADDLIB input-%s.a\n' "$$index" >> "$$staging/merge.mri"; \
				index=$$((index + 1)); \
			done; \
			printf 'SAVE\nEND\n' >> "$$staging/merge.mri"; \
			(cd "$$staging" && $(AR) -M < merge.mri); \
			mv "$$staging/merged.a" "$(prefix)/lib/libruy.a" ;; \
		*) echo "Unsupported archive platform: $(OS_NAME)" >&2; exit 1 ;; \
	esac; \
	test -s "$(prefix)/lib/libruy.a"; \
	echo "Installed libruy.a to $(prefix)/lib/"
