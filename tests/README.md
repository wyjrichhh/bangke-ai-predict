# Capture-only initialization trace tests

This target compiles the diagnostic plugin sources and reuses compatible Rime,
Boost, static glog and CT2 dependencies. It does not rebuild Rime and does not
replace its `BUILD_TEST=ON` tests or Windows/frontend acceptance.

```sh
cmake -S tests -B <task-build> -DCMAKE_BUILD_TYPE=Release \
  -DRIME_ROOT=<librime-source-and-dependency-prefix> \
  -DRIME_BUILD_ROOT=<configured-librime-build> \
  -DCTRANSLATE2_ROOT=<matching-ct2-installation> \
  -DAI_TEST_RIME_LIBRARY=<matching-rime-library> \
  -DAI_TEST_GLOG_LIBRARY=<matching-static-glog-library> \
  -DAI_TEST_BOOST_INCLUDE=<boost-headers>
cmake --build <task-build> --config Release --target predict_trace_test
ctest --test-dir <task-build> -C Release --output-on-failure
```

Windows must use the architecture/MT runtime matching the engine and explicitly
supply its actual Release `rime.lib`; the default `build/lib` search is not a
promise about Windows output layouts.

The Release test attaches an in-memory `google::LogSink`, verifies ERROR stage,
exception kind, numeric system code/category and JSON ID, bounded/control-free
path-redacted messages, and original exception type/what/code after rethrow.
It calls Query twice to ensure capture adds no failure latch or fail-open state
cleanup. CT2 keeps its original false return for a broken model.

On Windows this fixture directs the Rime DLL's private glog to stderr because
its log handles live until process exit; the EXE's in-memory Capture sink still
checks trace records. Actual Windows ERROR file persistence is verified by the
separate CAPI producer and installation acceptance, not by this fixture.

Optionally set `-DBANGKE_AI_TEST_MODEL_DIR=<existing-known-working-model>` to add
`predict_trace_model`, which reads the existing model without copying or changing
it and verifies CPU initialization and one prediction plus completion traces.
This diagnostic branch intentionally retains the original path handling: a
previously broken Windows Unicode model path may remain broken and should be
investigated using the captured exception rather than expected to pass.

Only new `ai_predict_init` records use the bounded redaction helper. Technical
`what` messages are retained only for the fixed model-initialization stage
allowlist; other runtime messages are omitted while type/code metadata remains.
Device values are emitted only as `cpu`, `cuda` or `other`. Windows failure
records additionally sample `GlobalMemoryStatusEx` physical/commit availability
and its success/error result; other platforms explicitly mark it unsupported. Existing
ordinary INFO messages may still contain input/context/model paths. They are not
covered by a claim that all historical logs are sanitized. Customer exception
capture and release ERROR file output still require the actual Windows build.
