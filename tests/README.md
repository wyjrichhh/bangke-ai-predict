# Unicode model initialization and fail-open tests

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
path-redacted messages. Query failures return no AI translation and clear pending
and AI text independently, allowing the subsequent echo translator to produce a
candidate (or no candidate in AI-only mode). This checks the translator chain,
not a complete Chinese dictionary or frontend UI.

On Windows this fixture directs the Rime DLL's private glog to stderr because
its log handles live until process exit; the EXE's in-memory Capture sink still
checks trace records. Actual Windows ERROR file persistence is verified by the
separate CAPI producer and installation acceptance, not by this fixture.

Optionally set `-DBANGKE_AI_TEST_MODEL_DIR=<existing-known-working-model>` to add
`predict_trace_model`, which creates a Chinese/space/~ model directory in its
isolated temporary directory, hard-links model.bin and copies small metadata,
then verifies CPU initialization and one prediction. The source model is never
modified. Source and temporary directory must be on the same filesystem for the
hard link. Windows acceptance must also exercise the installed input method in
the existing Chinese account and its real Chinese dictionary.

Initialization ERROR records use the bounded redaction helper. Per-stage and
per-key INFO and input/prediction/candidate bodies are not emitted. Technical
`what` messages are retained only for the fixed model-initialization stage
allowlist; other runtime messages are omitted while type/code metadata remains.
Device values are emitted only as `cpu`, `cuda` or `other`. Windows failure
records additionally sample `GlobalMemoryStatusEx` physical/commit availability
and its success/error result; other platforms explicitly mark it unsupported.
The plugin no longer logs input, context, prediction text or full model paths at
the reviewed call sites. Historical logs and other host components are outside
this guarantee. Customer exception
capture and release ERROR file output still require the actual Windows build.

The real-model fixture explicitly releases the EXE's shared backend, verifies
that its weak reference expires, rejects new loads during finalization, and
reopens the lifecycle for another real prediction. Release joins background
teardown threads, including their CRT/TLS exit. The Rime DLL has separate plugin
state; its module-finalize hook must additionally pass an isolated CAPI process
exit test with an external timeout. The fixture does not prove DLL unload safety
by testing the EXE's state alone.
