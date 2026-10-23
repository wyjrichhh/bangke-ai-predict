// Copyright RIME Developers
// Distributed under the BSD License

#include <glog/logging.h>

#include <iostream>
#include <stdexcept>

#include "ct2_backend.h"
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  google::InitGoogleLogging(argv[0]);
  rime::predict::InferenceBackendConfig config;
  config.model_path = argv[1];
  config.device = "cpu";
  for (int cycle = 0; cycle < 3; ++cycle) {
    auto backend = rime::predict::CT2Backend::Shared(config);
    if (!backend || !backend->IsInitialized())
      throw std::runtime_error("Model initialization failed");
    if (rime::predict::CT2Backend::Shared(config) != backend)
      throw std::runtime_error("Cache reuse failed");
    auto result = backend->Predict("jintiantianqihenhao", 64);
    if (!result || result->find("今天天气很好") == std::string::npos)
      throw std::runtime_error("Prediction failed");
    if (cycle == 1) google::ShutdownGoogleLogging();
    rime::predict::CT2Backend::ReleaseShared();
    rime::predict::CT2Backend::ReleaseShared();
    if (backend->IsInitialized())
      throw std::runtime_error("Released backend remains initialized");
    if (cycle == 1) google::InitGoogleLogging(argv[0]);
    std::cout << "cycle " << cycle + 1 << ": reuse/inference/release PASS\n";
  }
  google::ShutdownGoogleLogging();
}
