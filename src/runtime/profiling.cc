/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/profiling.cc
 * \brief Runtime profiling including timers.
 */

#include <dmlc/json.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/threading_backend.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <thread>

// kyunam
#include <fstream> 
#include <unistd.h>     // pipe, fork, dup2, execl, usleep, close
#include <sys/types.h>
#include <sys/wait.h>   // waitpid
#include <sys/stat.h>   // stat
#include <csignal>      // kill, SIGINT
#include <vector>
#include <string>
#include <cstdlib>      // std::stod, std::getenv
#include <cmath> 

namespace tvm {
namespace runtime {

class DefaultTimerNode : public TimerNode {
 public:
  virtual void Start() {
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    start_ = std::chrono::high_resolution_clock::now();
  }
  virtual void Stop() {
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    duration_ = std::chrono::high_resolution_clock::now() - start_;
  }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~DefaultTimerNode() {}

  explicit DefaultTimerNode(Device dev) : device_(dev) {}
  static constexpr const char* _type_key = "DefaultTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(DefaultTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
  Device device_;
};

TVM_REGISTER_OBJECT_TYPE(DefaultTimerNode);
TVM_REGISTER_OBJECT_TYPE(TimerNode);

Timer DefaultTimer(Device dev) { return Timer(make_object<DefaultTimerNode>(dev)); }

class CPUTimerNode : public TimerNode {
 public:
  virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~CPUTimerNode() {}

  static constexpr const char* _type_key = "CPUTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CPUTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};
TVM_REGISTER_OBJECT_TYPE(CPUTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.cpu").set_body_typed([](Device dev) {
  return Timer(make_object<CPUTimerNode>());
});

// keep track of which timers are not defined but we have already warned about
std::set<DLDeviceType> seen_devices;
std::mutex seen_devices_lock;

Timer Timer::Start(Device dev) {
  auto f = Registry::Get(std::string("profiling.timer.") + DLDeviceType2Str(dev.device_type));
  if (f == nullptr) {
    {
      std::lock_guard<std::mutex> lock(seen_devices_lock);
      if (seen_devices.find(dev.device_type) == seen_devices.end()) {
        LOG(WARNING)
            << "No timer implementation for " << DLDeviceType2Str(dev.device_type)
            << ", using default timer instead. It may be inaccurate or have extra overhead.";
        seen_devices.insert(dev.device_type);
      }
    }
    Timer t = DefaultTimer(dev);
    t->Start();
    return t;
  } else {
    Timer t = f->operator()(dev);
    t->Start();
    return t;
  }
}

TVM_REGISTER_GLOBAL("profiling.start_timer").set_body_typed(Timer::Start);

namespace profiling {

Profiler::Profiler(std::vector<Device> devs, std::vector<MetricCollector> metric_collectors,
                   std::unordered_map<String, ObjectRef> configuration)
    : devs_(devs), collectors_(metric_collectors), configuration_(configuration) {
  is_running_ = false;
  std::vector<DeviceWrapper> wrapped_devs;
  for (auto dev : devs) {
    wrapped_devs.push_back(DeviceWrapper(make_object<DeviceWrapperNode>(dev)));
  }
  for (auto& x : collectors_) {
    x->Init(wrapped_devs);
  }
  // reset the thread pool so that PAPI eventset hooks are set in all threads.
  threading::ResetThreadPool();

  configuration_[String("Number of threads")] =
      ObjectRef(make_object<CountNode>(threading::NumThreads()));
}

void Profiler::Start() {
  is_running_ = true;
  for (auto dev : devs_) {
    StartCall("Total", dev, {});
  }
}

void Profiler::StartCall(String name, Device dev,
                         std::unordered_map<std::string, ObjectRef> extra_metrics) {
  std::vector<std::pair<MetricCollector, ObjectRef>> objs;
  for (auto& collector : collectors_) {
    ObjectRef obj = collector->Start(dev);
    if (obj.defined()) {
      objs.emplace_back(collector, obj);
    }
  }
  in_flight_.push(CallFrame{dev, name, Timer::Start(dev), extra_metrics, objs});
}

void Profiler::StopCall(std::unordered_map<std::string, ObjectRef> extra_metrics) {
  CallFrame cf = in_flight_.top();
  cf.timer->Stop();
  for (auto& p : extra_metrics) {
    cf.extra_metrics[p.first] = p.second;
  }
  // collect the extra metrics from user defined collectors
  for (const auto& obj : cf.extra_collectors) {
    auto collector_metrics = obj.first->Stop(obj.second);
    for (auto& p : collector_metrics) {
      cf.extra_metrics[p.first] = p.second;
    }
  }
  in_flight_.pop();
  calls_.push_back(cf);
}

void Profiler::Stop() {
  is_running_ = false;
  for (size_t i = 0; i < devs_.size(); i++) {
    StopCall();
  }
}

std::vector<int64_t> ToShape(NDArray shape_tensor) {
  std::vector<int64_t> shape;
  auto rank = shape_tensor.Shape().size();
  auto dtype = shape_tensor.DataType();

  // For 0-rank shapes we need to allocate a single scalar.
  if (rank == 0) {
    return shape;
  }

  // Otherwise we should be rank-1, and we will extract the number of dimensions
  // for the output vector.
  ICHECK_EQ(rank, 1U) << "shape tensor should be a k-length vector, found " << rank;
  int64_t ndim = shape_tensor.Shape().at(0);
  shape.resize(ndim);

  const DLTensor* dl_tensor = shape_tensor.operator->();
  if (dtype.is_int() && dtype.bits() == 32 && dtype.lanes() == 1) {
    int32_t* dims = reinterpret_cast<int32_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else if (dtype.is_int() && dtype.bits() == 64 && dtype.lanes() == 1) {
    int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else {
    LOG(FATAL) << "invalid shape tensor datatype: " << dtype;
  }

  return shape;
}

String ShapeString(NDArray shape, DLDataType dtype) { return ShapeString(ToShape(shape), dtype); }

String ShapeString(const std::vector<int64_t>& shape, DLDataType dtype) {
  std::stringstream sizes;
  sizes << dtype << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) {
      sizes << ", ";
    }
    sizes << shape[i];
  }
  sizes << "]";
  return String(sizes.str());
}

String ShapeString(const std::vector<NDArray>& shapes) {
  std::stringstream sizes;
  for (const NDArray& ary : shapes) {
    if (sizes.tellp() > 0) {
      sizes << ", ";
    }
    auto shape = ary.Shape();
    sizes << ary.DataType() << "[";
    for (size_t i = 0; i < shape.size(); i++) {
      if (i != 0) {
        sizes << ", ";
      }
      sizes << shape[i];
    }
    sizes << "]";
  }
  return String(sizes.str());
}

String ReportNode::AsCSV() const {
  // get unique headers
  std::set<std::string> unique_headers;

  for (auto row : calls) {
    for (auto p : row) {
      unique_headers.insert(p.first);
    }
  }

  std::vector<std::string> headers;
  for (auto x : unique_headers) {
    headers.push_back(x);
  }

  std::stringstream s;

  for (size_t i = 0; i < headers.size(); i++) {
    std::string header = headers[i];
    s << header;
    if (i < headers.size() - 1) {
      s << ",";
    }
  }
  s << std::endl;
  for (auto row : calls) {
    for (size_t i = 0; i < headers.size(); i++) {
      std::string header = headers[i];
      auto it = row.find(header);
      if (it != row.end()) {
        std::string val;
        if ((*it).second.as<CountNode>()) {
          s << (*it).second.as<CountNode>()->value;
        } else if ((*it).second.as<DurationNode>()) {
          s << (*it).second.as<DurationNode>()->microseconds;
        } else if ((*it).second.as<PercentNode>()) {
          s << (*it).second.as<PercentNode>()->percent;
        } else if ((*it).second.as<RatioNode>()) {
          s << (*it).second.as<RatioNode>()->ratio;
        } else if ((*it).second.as<StringObj>()) {
          s << "\"" << Downcast<String>((*it).second) << "\"";
        }
      }
      if (i < headers.size() - 1) {
        s << ",";
      }
    }
    s << std::endl;
  }
  return s.str();
}

namespace {
void metric_as_json(std::ostream& os, ObjectRef o) {
  if (o.as<StringObj>()) {
    os << "{\"string\":"
       << "\"" << Downcast<String>(o) << "\""
       << "}";
  } else if (const CountNode* n = o.as<CountNode>()) {
    os << "{\"count\":" << n->value << "}";
  } else if (const DurationNode* n = o.as<DurationNode>()) {
    os << "{\"microseconds\":" << std::setprecision(std::numeric_limits<double>::max_digits10)
       << std::fixed << n->microseconds << "}";
  } else if (const PercentNode* n = o.as<PercentNode>()) {
    os << "{\"percent\":" << std::setprecision(std::numeric_limits<double>::max_digits10)
       << std::fixed << n->percent << "}";
  } else if (const RatioNode* n = o.as<RatioNode>()) {
    os << "{\"ratio\":" << std::setprecision(std::numeric_limits<double>::max_digits10)
       << std::fixed << n->ratio << "}";
  } else {
    LOG(FATAL) << "Unprintable type " << o->GetTypeKey();
  }
}
}  // namespace

String ReportNode::AsJSON() const {
  std::ostringstream s;
  // DMLC's JSONWriter does not allow us to write a key value pair without
  // implementing Write for the value. We want a specific write for the value,
  // so we would have to implement a custom data structure for each type of
  // value we want to print. Instead we construct the json by hand because it
  // is easier.
  s << "{";

  s << "\"calls\":[";
  for (size_t i = 0; i < calls.size(); i++) {
    size_t j = 0;
    s << "{";
    for (const auto& kv : calls[i]) {
      s << "\"" << kv.first << "\":";
      metric_as_json(s, kv.second);
      if (j < calls[i].size() - 1) {
        s << ",";
      }
      j++;
    }
    s << "}";
    if (i < calls.size() - 1) {
      s << ",";
    }
  }
  s << "],";  // end calls

  s << "\"device_metrics\":{";
  size_t i = 0;
  for (const auto& dev_kv : device_metrics) {
    size_t j = 0;
    s << "\"" << dev_kv.first << "\":{";
    for (const auto& metric_kv : dev_kv.second) {
      s << "\"" << metric_kv.first << "\":";
      metric_as_json(s, metric_kv.second);
      if (j < dev_kv.second.size() - 1) {
        s << ",";
      }
      j++;
    }
    s << "}";
    if (i < device_metrics.size() - 1) {
      s << ",";
    }
    i++;
  }
  s << "},";  // end device metrics

  s << "\"configuration\":{";
  size_t k = 0;
  for (const auto& kv : configuration) {
    s << "\"" << kv.first << "\":";
    metric_as_json(s, kv.second);
    if (k < configuration.size() - 1) {
      s << ",";
    }
    k++;
  }
  s << "}";  // end configuration
  s << "}";
  return s.str();
}

// Aggregate a set of values for a metric. Computes sum for Duration, Count,
// and Percent; average for Ratio; and assumes all Strings are the same. All
// ObjectRefs in metrics must have the same type.
ObjectRef AggregateMetric(const std::vector<ObjectRef>& metrics) {
  ICHECK_GT(metrics.size(), 0) << "Must pass a non-zero number of metrics";
  if (metrics[0].as<DurationNode>()) {
    double sum = 0;
    for (auto& metric : metrics) {
      sum += metric.as<DurationNode>()->microseconds;
    }
    return ObjectRef(make_object<DurationNode>(sum));
  } else if (metrics[0].as<CountNode>()) {
    int64_t sum = 0;
    for (auto& metric : metrics) {
      sum += metric.as<CountNode>()->value;
    }
    return ObjectRef(make_object<CountNode>(sum));
  } else if (metrics[0].as<PercentNode>()) {
    double sum = 0;
    for (auto& metric : metrics) {
      sum += metric.as<PercentNode>()->percent;
    }
    return ObjectRef(make_object<PercentNode>(sum));
  } else if (metrics[0].as<RatioNode>()) {
    double sum = 0;
    for (auto& metric : metrics) {
      sum += metric.as<RatioNode>()->ratio;
    }
    return ObjectRef(make_object<RatioNode>(sum / metrics.size()));
  } else if (metrics[0].as<StringObj>()) {
    for (auto& m : metrics) {
      if (Downcast<String>(metrics[0]) != Downcast<String>(m)) {
        return ObjectRef(String(""));
      }
    }
    // Assume all strings in metrics are the same.
    return metrics[0];
  } else {
    LOG(FATAL) << "Can only aggregate metrics with types DurationNode, CountNode, "
                  "PercentNode, RatioNode, and StringObj, but got "
               << metrics[0]->GetTypeKey();
    return ObjectRef();  // To silence warnings
  }
}

// Try and set the locale of the provided stringstream so that it will print
// numbers with thousands separators. Sometimes users will have a misconfigured
// system where an invalid locale is set, so we catch and ignore any locale
// errors.
static void set_locale_for_separators(std::stringstream& s) {
  try {
    // empty string indicates locale should be the user's default, see man 3 setlocale
    s.imbue(std::locale(""));
  } catch (std::runtime_error& e) {
  }
}

static String print_metric(ObjectRef metric) {
  std::string val;
  if (metric.as<CountNode>()) {
    std::stringstream s;
    set_locale_for_separators(s);
    s << std::fixed << metric.as<CountNode>()->value;
    val = s.str();
  } else if (metric.as<DurationNode>()) {
    std::stringstream s;
    set_locale_for_separators(s);
    s << std::fixed << std::setprecision(2) << metric.as<DurationNode>()->microseconds;
    val = s.str();
  } else if (metric.as<PercentNode>()) {
    std::stringstream s;
    s << std::fixed << std::setprecision(2) << metric.as<PercentNode>()->percent;
    val = s.str();
  } else if (metric.as<RatioNode>()) {
    std::stringstream s;
    set_locale_for_separators(s);
    s << std::setprecision(2) << metric.as<RatioNode>()->ratio;
    val = s.str();
  } else if (metric.as<StringObj>()) {
    val = Downcast<String>(metric);
  } else {
    LOG(FATAL) << "Cannot print metric of type " << metric->GetTypeKey();
  }
  return val;
}

String ReportNode::AsTable(bool sort, bool aggregate, bool compute_col_sums) const {
  // aggregate calls by op hash (or op name if hash is not set) + argument shapes
  std::vector<Map<String, ObjectRef>> aggregated_calls;
  if (aggregate) {
    std::unordered_map<std::string, std::vector<size_t>> aggregates;
    for (size_t i = 0; i < calls.size(); i++) {
      auto& frame = calls[i];
      auto it = frame.find("Hash");
      std::string name = Downcast<String>(frame["Name"]);
      if (it != frame.end()) {
        name = Downcast<String>((*it).second);
      }
      if (frame.find("Argument Shapes") != frame.end()) {
        name += Downcast<String>(frame["Argument Shapes"]);
      }
      if (frame.find("Device") != frame.end()) {
        name += Downcast<String>(frame["Device"]);
      }

      if (aggregates.find(name) == aggregates.end()) {
        aggregates[name] = {i};
      } else {
        aggregates[name].push_back(i);
      }
    }
    for (const auto& p : aggregates) {
      std::unordered_map<String, ObjectRef> aggregated;
      std::unordered_set<std::string> metrics;
      for (auto& call : calls) {
        for (auto& metric : call) {
          metrics.insert(metric.first);
        }
      }
      for (const std::string& metric : metrics) {
        std::vector<ObjectRef> per_call;
        for (auto i : p.second) {
          auto& call = calls[i];
          auto it = std::find_if(call.begin(), call.end(),
                                 [&metric](const std::pair<String, ObjectRef>& call_metric) {
                                   return std::string(call_metric.first) == metric;
                                 });
          if (it != call.end()) {
            per_call.push_back((*it).second);
          }
        }
        if (per_call.size() > 0) {
          aggregated[metric] = AggregateMetric(per_call);
        }
      }
      aggregated_calls.push_back(aggregated);
    }
  } else {
    for (auto call : calls) {
      aggregated_calls.push_back(call);
    }
  }

  // sort rows by duration
  if (sort) {
    std::sort(aggregated_calls.begin(), aggregated_calls.end(),
              [&](const Map<String, ObjectRef>& a, const Map<String, ObjectRef>& b) {
                return a.at("Duration (us)").as<DurationNode>()->microseconds >
                       b.at("Duration (us)").as<DurationNode>()->microseconds;
              });
  }

  // compute columnwise sums
  if (compute_col_sums) {
    std::unordered_map<String, ObjectRef> col_sums;
    for (auto call : aggregated_calls) {
      for (auto p : call) {
        if (p.second.as<CountNode>()) {
          int64_t val = p.second.as<CountNode>()->value;
          auto it = col_sums.find(p.first);
          if (it != col_sums.end()) {
            val += it->second.as<CountNode>()->value;
          }
          col_sums[p.first] = ObjectRef(make_object<CountNode>(val));
        } else if (p.second.as<DurationNode>()) {
          double val = p.second.as<DurationNode>()->microseconds;
          auto it = col_sums.find(p.first);
          if (it != col_sums.end()) {
            val += it->second.as<DurationNode>()->microseconds;
          }
          col_sums[p.first] = ObjectRef(make_object<DurationNode>(val));
        } else if (p.second.as<PercentNode>()) {
          double val = p.second.as<PercentNode>()->percent;
          auto it = col_sums.find(p.first);
          if (it != col_sums.end()) {
            val += it->second.as<PercentNode>()->percent;
          }
          col_sums[p.first] = ObjectRef(make_object<PercentNode>(val));
        } else if (p.second.as<RatioNode>()) {
          // It does not make sense to sum ratios
        }
      }
    }
    col_sums["Name"] = String("Sum");
    aggregated_calls.push_back({{String("Name"), String("----------")}});  // separator
    aggregated_calls.push_back(col_sums);
  }

  // per-device metrics
  for (auto p : device_metrics) {
    Map<String, ObjectRef> metrics = p.second;
    metrics.Set("Name", String("Total"));
    aggregated_calls.push_back(metrics);
  }

  // Table formatting
  std::set<std::string> unique_headers;
  for (auto row : aggregated_calls) {
    for (auto p : row) {
      unique_headers.insert(p.first);
    }
  }

  // always include these headers in this order
  std::vector<std::string> headers = {"Name",   "Duration (us)", "Percent",
                                      "Device", "Count",         "Argument Shapes"};
  for (auto header : unique_headers) {
    if (std::find(headers.begin(), headers.end(), header) == headers.end()) {
      headers.push_back(header);
    }
  }

  // Switch layout from row major to column major so we can easily compute column widths.
  std::vector<std::vector<std::string>> cols;
  for (auto header : headers) {
    cols.push_back({header});
  }
  for (auto row : aggregated_calls) {
    for (size_t i = 0; i < headers.size(); i++) {
      auto it = row.find(headers[i]);
      if (it == row.end()) {
        // fill empty data with empty strings
        cols[i].push_back("");
      } else {
        cols[i].push_back(print_metric((*it).second));
      }
    }
  }

  std::vector<size_t> widths;
  for (auto v : cols) {
    size_t width = 0;
    for (auto x : v) {
      width = std::max(width, x.size());
    }
    widths.push_back(width);
  }
  size_t length = 0;
  for (auto v : cols) {
    length = std::max(length, v.size());
  }

  std::stringstream s;
  for (size_t row = 0; row < length; row++) {
    for (size_t col = 0; col < cols.size(); col++) {
      // left align first column
      if (col == 0) {
        s << std::left;
      } else {
        s << std::right;
      }
      if (row < cols[col].size()) {
        s << std::setw(widths[col]) << cols[col][row] << "  ";
      } else {
        s << std::setw(widths[col]) << ""
          << "  ";
      }
    }
    s << std::endl;
  }

  // Add configuration information. It will not be aligned with the columns.
  s << std::endl << "Configuration" << std::endl << "-------------" << std::endl;
  for (auto kv : configuration) {
    s << kv.first << ": " << print_metric(kv.second) << std::endl;
  }
  return s.str();
}

std::string DeviceString(Device dev) {
  return DLDeviceType2Str(dev.device_type) + std::to_string(dev.device_id);
}

Report Profiler::Report() {
  // sync all timers and normalize rows
  std::vector<std::unordered_map<String, ObjectRef>> rows;
  for (auto& cf : calls_) {
    std::unordered_map<String, ObjectRef> row;
    double us = cf.timer->SyncAndGetElapsedNanos() / 1e3;
    row["Duration (us)"] = ObjectRef(make_object<DurationNode>(us));
    row["Count"] = ObjectRef(make_object<CountNode>(1));
    row["Name"] = cf.name;
    row["Device"] = String(DeviceString(cf.dev));
    for (auto p : cf.extra_metrics) {
      row[p.first] = p.second;
    }
    rows.push_back(row);
  }

  // the last couple of call frames are the overall times
  double overall_time_us = 0;
  std::unordered_map<String, Map<String, ObjectRef>> device_metrics;
  for (size_t i = 0; i < devs_.size(); i++) {
    auto row = rows[rows.size() - 1];
    rows.pop_back();
    device_metrics[Downcast<String>(row["Device"])] = row;
    overall_time_us =
        std::max(overall_time_us, row["Duration (us)"].as<DurationNode>()->microseconds);
  }

  // Calculate percentages
  for (auto& row : rows) {
    row["Percent"] = ObjectRef(make_object<PercentNode>(
        row["Duration (us)"].as<DurationNode>()->microseconds / overall_time_us * 100));
  }

  // convert to map
  std::vector<Map<String, ObjectRef>> converted_rows;
  for (const auto& row : rows) {
    converted_rows.push_back(row);
  }

  return profiling::Report(converted_rows, device_metrics, configuration_);
}

Report::Report(Array<Map<String, ObjectRef>> calls,
               Map<String, Map<String, ObjectRef>> device_metrics,
               Map<String, ObjectRef> configuration) {
  auto node = make_object<ReportNode>();
  node->calls = std::move(calls);
  node->device_metrics = std::move(device_metrics);
  node->configuration = std::move(configuration);
  data_ = std::move(node);
}

Map<String, ObjectRef> parse_metrics(dmlc::JSONReader* reader) {
  reader->BeginObject();
  std::string metric_name, metric_value_name;
  Map<String, ObjectRef> metrics;
  while (reader->NextObjectItem(&metric_name)) {
    ObjectRef o;
    reader->BeginObject();
    reader->NextObjectItem(&metric_value_name);
    if (metric_value_name == "microseconds") {
      double microseconds;
      reader->Read(&microseconds);
      o = ObjectRef(make_object<DurationNode>(microseconds));
    } else if (metric_value_name == "percent") {
      double percent;
      reader->Read(&percent);
      o = ObjectRef(make_object<PercentNode>(percent));
    } else if (metric_value_name == "count") {
      int64_t count;
      reader->Read(&count);
      o = ObjectRef(make_object<CountNode>(count));
    } else if (metric_value_name == "ratio") {
      double ratio;
      reader->Read(&ratio);
      o = ObjectRef(make_object<RatioNode>(ratio));
    } else if (metric_value_name == "string") {
      std::string s;
      reader->Read(&s);
      o = String(s);
    } else {
      LOG(FATAL) << "Cannot parse metric of type " << metric_value_name
                 << " valid types are microseconds, percent, count.";
    }
    metrics.Set(metric_name, o);
    // Necessary to make sure that the parser hits the end of the object.
    ICHECK(!reader->NextObjectItem(&metric_value_name));
    // EndObject does not exist, leaving this here for clarity
    // reader.EndObject();
  }
  // reader.EndObject();
  return metrics;
}

Report Report::FromJSON(String json) {
  std::stringstream input(json.operator std::string());
  dmlc::JSONReader reader(&input);
  std::string key;
  Array<Map<String, ObjectRef>> calls;
  Map<String, Map<String, ObjectRef>> device_metrics;
  Map<String, ObjectRef> configuration;

  reader.BeginObject();
  while (reader.NextObjectItem(&key)) {
    if (key == "calls") {
      reader.BeginArray();
      while (reader.NextArrayItem()) {
        calls.push_back(parse_metrics(&reader));
      }
      // reader.EndArray();
    } else if (key == "device_metrics") {
      reader.BeginObject();
      std::string device_name;
      while (reader.NextObjectItem(&device_name)) {
        device_metrics.Set(device_name, parse_metrics(&reader));
      }
      // reader.EndObject();
    } else if (key == "configuration") {
      configuration = parse_metrics(&reader);
    }
  }

  return Report(calls, device_metrics, configuration);
}

TVM_REGISTER_OBJECT_TYPE(DurationNode);
TVM_REGISTER_OBJECT_TYPE(PercentNode);
TVM_REGISTER_OBJECT_TYPE(CountNode);
TVM_REGISTER_OBJECT_TYPE(RatioNode);
TVM_REGISTER_OBJECT_TYPE(ReportNode);
TVM_REGISTER_OBJECT_TYPE(DeviceWrapperNode);
TVM_REGISTER_OBJECT_TYPE(MetricCollectorNode);

TVM_REGISTER_GLOBAL("runtime.profiling.AsTable").set_body_method<Report>(&ReportNode::AsTable);
TVM_REGISTER_GLOBAL("runtime.profiling.AsCSV").set_body_typed([](Report n) { return n->AsCSV(); });
TVM_REGISTER_GLOBAL("runtime.profiling.AsJSON").set_body_typed([](Report n) {
  return n->AsJSON();
});
TVM_REGISTER_GLOBAL("runtime.profiling.FromJSON").set_body_typed(Report::FromJSON);
TVM_REGISTER_GLOBAL("runtime.profiling.DeviceWrapper").set_body_typed([](Device dev) {
  return DeviceWrapper(dev);
});

PackedFunc ProfileFunction(Module mod, std::string func_name, int device_type, int device_id,
                           int warmup_iters, Array<MetricCollector> collectors) {
  // Module::GetFunction is not const, so this lambda has to be mutable
  return PackedFunc([=](TVMArgs args, TVMRetValue* ret) mutable {
    PackedFunc f = mod.GetFunction(func_name);
    CHECK(f.defined()) << "There is no function called \"" << func_name << "\" in the module";
    Device dev{static_cast<DLDeviceType>(device_type), device_id};

    // warmup
    for (int i = 0; i < warmup_iters; i++) {
      f.CallPacked(args, ret);
    }

    for (auto& collector : collectors) {
      collector->Init({DeviceWrapper(dev)});
    }
    std::vector<Map<String, ObjectRef>> results;
    results.reserve(collectors.size());
    std::vector<std::pair<MetricCollector, ObjectRef>> collector_data;
    collector_data.reserve(collectors.size());
    for (auto& collector : collectors) {
      ObjectRef o = collector->Start(dev);
      // If not defined, then the collector cannot time this device.
      if (o.defined()) {
        collector_data.push_back({collector, o});
      }
    }

    // TODO(tkonolige): repeated calls if the runtime is small?
    f.CallPacked(args, ret);

    for (auto& kv : collector_data) {
      results.push_back(kv.first->Stop(kv.second));
    }
    Map<String, ObjectRef> combined_results;
    for (auto m : results) {
      for (auto p : m) {
        // assume that there is no shared metric name between collectors
        combined_results.Set(p.first, p.second);
      }
    }
    *ret = combined_results;
  });
}

TVM_REGISTER_GLOBAL("runtime.profiling.ProfileFunction")
    .set_body_typed<PackedFunc(Module, String, int, int, int,
                               Array<MetricCollector>)>([](Module mod, String func_name,
                                                           int device_type, int device_id,
                                                           int warmup_iters,
                                                           Array<MetricCollector> collectors) {
      if (mod->type_key() == std::string("rpc")) {
        LOG(FATAL)
            << "Profiling a module over RPC is not yet supported";  // because we can't send
                                                                    // MetricCollectors over rpc.
        throw;
      } else {
        return ProfileFunction(mod, func_name, device_type, device_id, warmup_iters, collectors);
      }
    });

PackedFunc WrapTimeEvaluator(PackedFunc pf, Device dev, int number, int repeat, int min_repeat_ms,
                             int limit_zero_time_iterations, int cooldown_interval_ms,
                             int repeats_to_cooldown, int cache_flush_bytes, PackedFunc f_preproc) {
  // std::cout << "<profiling.cc> WrapTimeEvaluator is called" << std::endl; // kyunam

  ICHECK(pf != nullptr);

  // std::cout << "<profiling.cc> device_type is " << static_cast<int>(dev.device_type) << std::endl; // kyunam
  // 1: kDLCPU, 2: kDLCUDA, 3: kDLCUDAHost, ... // kyunam

  if (static_cast<int>(dev.device_type) == static_cast<int>(kDLMicroDev)) {
    auto get_micro_time_evaluator = runtime::Registry::Get("micro._GetMicroTimeEvaluator");
    ICHECK(get_micro_time_evaluator != nullptr) << "micro backend not enabled";
    return (*get_micro_time_evaluator)(pf, dev, number, repeat);
  }

  // std::cout << "<profiling.cc> ftimer as a lambda function is being defined and returned to python" << std::endl; // kyunam

  // Original ftimer
  // auto ftimer = [pf, dev, number, repeat, min_repeat_ms, limit_zero_time_iterations,
  //                cooldown_interval_ms, repeats_to_cooldown, cache_flush_bytes,
  //                f_preproc](TVMArgs args, TVMRetValue* rv) mutable {
  //   TVMRetValue temp;
  //   std::ostringstream os;
  //   // skip first time call, to activate lazy compilation components.
  //   pf.CallPacked(args, &temp);

  //   // allocate two large arrays to flush L2 cache
  //   NDArray arr1, arr2;
  //   if (cache_flush_bytes > 0) {
  //     arr1 = NDArray::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
  //     arr2 = NDArray::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
  //   }

  //   DeviceAPI::Get(dev)->StreamSync(dev, nullptr);

  //   for (int i = 0; i < repeat; ++i) {
  //     if (f_preproc != nullptr) {
  //       f_preproc.CallPacked(args, &temp);
  //     }
  //     double duration_ms = 0.0;
  //     int absolute_zero_times = 0;
  //     do {
  //       if (duration_ms > 0.0) {
  //         const double golden_ratio = 1.618;
  //         number = static_cast<int>(
  //             std::max((min_repeat_ms / (duration_ms / number) + 1), number * golden_ratio));
  //       }
  //       if (cache_flush_bytes > 0) {
  //         arr1.CopyFrom(arr2);
  //       }
  //       DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
  //       // start timing
  //       Timer t = Timer::Start(dev);
  //       for (int j = 0; j < number; ++j) {
  //         pf.CallPacked(args, &temp);
  //       }
  //       t->Stop();
  //       int64_t t_nanos = t->SyncAndGetElapsedNanos();
  //       if (t_nanos == 0) absolute_zero_times++;
  //       duration_ms = t_nanos / 1e6;
  //     } while (duration_ms < min_repeat_ms && absolute_zero_times < limit_zero_time_iterations);

  //     double speed = duration_ms / 1e3 / number;
  //     os.write(reinterpret_cast<char*>(&speed), sizeof(speed));

  //     if (cooldown_interval_ms > 0 && (i % repeats_to_cooldown) == 0) {
  //       std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_interval_ms));
  //     }
  //   }

  //   std::string blob = os.str();
  //   TVMByteArray arr;
  //   arr.size = blob.length();
  //   arr.data = blob.data();
  //   // return the time.
  //   *rv = arr;
  // };
  
  // Custom ftimer
  // kyunam
  auto ftimer = [pf, dev, number = 10, repeat, min_repeat_ms, limit_zero_time_iterations,
                 cooldown_interval_ms, repeats_to_cooldown, cache_flush_bytes,
                 f_preproc](TVMArgs args, TVMRetValue* rv) mutable -> void {

    TVMRetValue temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, &temp);

    // allocate two large arrays to flush L2 cache
    NDArray arr1, arr2;
    if (cache_flush_bytes > 0) {
      arr1 = NDArray::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
      arr2 = NDArray::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
    }
    
    // Sync device
    DeviceAPI::Get(dev)->StreamSync(dev, nullptr);

    for (int i = 0; i < repeat; ++i) {
      if (f_preproc != nullptr) {
        f_preproc.CallPacked(args, &temp);
      }
      double duration_ms = 0.0;
      int absolute_zero_times = 0;
      double energy_mJ = 0.0;
      bool kyunam_invalid = false;
      double power_W = 0.0;
      do {
        if (duration_ms > 0.0) {
          const double golden_ratio = 1.618;
          number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1), number * golden_ratio));
        }
        if (cache_flush_bytes > 0) {
          arr1.CopyFrom(arr2);
        }
        
        if (static_cast<int>(dev.device_type) == static_cast<int>(kDLCPU)) {
          // RAPL energy file path
          std::string file_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
          std::ifstream file;

          // Energy values to be read
          double energy_begin, energy_end;
          
          // Sync device
          DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
          
          // Start timer
          Timer t = Timer::Start(dev);

          // Read energy value
          file.open(file_path);
          file >> energy_begin;
          file.close();

          for (int j = 0; j < number; ++j) {
            pf.CallPacked(args, &temp);
          }

          // Read energy value again
          file.open(file_path);
          file >> energy_end;
          file.close();

          // Stop timer
          t->Stop();

          // Convert total time to ms
          int64_t t_nanos = t->SyncAndGetElapsedNanos();
          if (t_nanos == 0) absolute_zero_times++;
          duration_ms = t_nanos / 1e6;

          // Convert total energy to mJ
          energy_mJ = (energy_end - energy_begin) / 1e3;

          // Check if the calculated energy value is normal
          // If energy_mJ < 0, the RAPL energy file wrapped-around
          // --> Need to measure again
          // If energy_mJ ~= 0, the RAPL energy file has not been updated yet
          // --> Need to extend the measurement interval
          kyunam_invalid = (energy_mJ < 1);
        } else if (static_cast<int>(dev.device_type) == static_cast<int>(kDLCUDA)) {
          // Set min_repeat_ms to 1000 to ensure enough samples are collected
          min_repeat_ms = 1000;

          // Create a pipe to capture child process output.
          int fd[2];
          if (pipe(fd) == -1) std::cout << "Pipe creation failed.\n";

	        // Fork to create a child process.
          pid_t pid = fork();
          if (pid < 0) std::cout << "Fork failed.\n";

	        if (pid == 0) {
            // Child process:
            close(fd[0]);               // Close read end.
            dup2(fd[1], STDOUT_FILENO); // Redirect stdout to the pipe.
            close(fd[1]);               // Close original write descriptor.
            
            // Execute the supported power measurement tool in loop mode (runs until SIGINT is received)
            struct stat buf;
            char* path = "/usr/bin/nvidia-smi";
            bool nvidia_smi_exists = (stat(path, &buf) == 0);
            path = "/usr/bin/tegrastats";
            bool tegrastats_exists = (stat(path, &buf) == 0);

            if (nvidia_smi_exists) {
              execl("/usr/bin/nvidia-smi", "nvidia-smi",
              "--query-gpu", "power.draw.instant",
              "--format=csv,noheader,nounits",
              "--loop-ms=100",
              (char *)nullptr);
            } else if (tegrastats_exists) {
              const char *cmd =
                // exit when receiving SIGINT
                "trap 'exit' INT; "
              
                // locate the first matching hwmon directory for each sensor
                "for d in /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*; do DIR0=\"$d\"; break; done; "
                "for d in /sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*; do DIR1=\"$d\"; break; done; "

                // verify each expanded glob points to a directory
                "if [ ! -d $DIR0 ]; then echo 'Error: DIR0 not found' >&2; exit 1; fi; "
                "if [ ! -d $DIR1 ]; then echo 'Error: DIR1 not found' >&2; exit 1; fi; "
        
                // validate labels in DIR0
                "if [ \"$(<$DIR0/in1_label)\" != 'VDD_GPU_SOC' ]; then "
                    "echo 'Error: DIR0 in1_label != VDD_GPU_SOC' >&2; exit 1; "
                "fi; "
                "if [ \"$(<$DIR0/in2_label)\" != 'VDD_CPU_CV' ]; then "
                    "echo 'Error: DIR0 in2_label != VDD_CPU_CV' >&2; exit 1; "
                "fi; "
                "if [ \"$(<$DIR0/in3_label)\" != 'VIN_SYS_5V0' ]; then "
                    "echo 'Error: DIR0 in3_label != VIN_SYS_5V0' >&2; exit 1; "
                "fi; "
        
                // validate label in DIR1
                "if [ \"$(<$DIR1/in2_label)\" != 'VDDQ_VDD2_1V8AO' ]; then "
                    "echo 'Error: DIR1 in2_label != VDDQ_VDD2_1V8AO' >&2; exit 1; "
                "fi; "

                // sampling loop: every 100ms, read inputs, multiply & sum, divide by 1e6
                "while sleep .1; do "
                    "awk "
                    "-v v1=$(<$DIR0/in1_input) -v c1=$(<$DIR0/curr1_input) "
                    "-v v2=$(<$DIR0/in2_input) -v c2=$(<$DIR0/curr2_input) "
                    "-v v3=$(<$DIR0/in3_input) -v c3=$(<$DIR0/curr3_input) "
                    "-v v4=$(<$DIR1/in2_input) -v c4=$(<$DIR1/curr2_input) "
                    "'BEGIN{print (v1*c1 + v2*c2 + v3*c3 + v4*c4)/1e6}'; "
                "done";
              execl("/usr/bin/bash", "bash", "-c", cmd, (char *)nullptr);
            } else {
              std::cout << "No supported power measurement tool: nvidia-smi, tegrastats" << std::endl;
            }

            // Should not reach here.
            std::cout << "execl failed" << std::endl;
	          _exit(127); 
          }

	        // Parent process:
          close(fd[1]); // Close write end; we only read in parent.

	        // Start a reader thread to capture output lines.
          std::vector<std::string> lines;
	        std::thread reader([&]() {
            constexpr size_t bufferSize = 10;
            char buffer[bufferSize];
            // Open a FILE* stream for easier line-based reading.
            FILE* stream = fdopen(fd[0], "r");
            if (!stream) return;
            while (fgets(buffer, bufferSize, stream) != nullptr) {
              std::string s(buffer);
              // Remove any trailing newline.
              if (!s.empty() && s.back() == '\n') {
                s.pop_back();
              }
              lines.push_back(s);
            }
            fclose(stream);
          });

          DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
          // start timing
          Timer t = Timer::Start(dev);
          for (int j = 0; j < number; ++j) {
            pf.CallPacked(args, &temp);
          }
          t->Stop();
          int64_t t_nanos = t->SyncAndGetElapsedNanos();
          if (t_nanos == 0) absolute_zero_times++;
          duration_ms = t_nanos / 1e6;

          // After work is finished and the child is alive, send SIGINT (Ctrl+C) to the child process.
          if (duration_ms < 300) usleep((300 - duration_ms) * 1000);
          kill(pid, SIGINT);
          // Wait for the child process to terminate.
          int status;
          waitpid(pid, &status, 0);

          // Wait for the reader thread to finish reading.
          reader.join();

	        // Check if we collected enough lines (need at least 8 to have meaningful average data).
          if (lines.size() <= 7) {
            kyunam_invalid = true;
	        } else {
            // Discard the first five and last two lines, then average the rest.
            double sum = 0;
            int count = 0;
            for (size_t i = 5; i < lines.size() - 2; ++i) {
              try {
                sum += std::stod(lines[i]);
                ++count;
              } catch (...) {
                // Skip conversion errors.
              }
            }
            if (count > 0) {
              power_W = sum / count;
              kyunam_invalid = false;
            } else {
              kyunam_invalid = true;
            }
	        }
        } else {
          DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
          // start timing
          Timer t = Timer::Start(dev);
          for (int j = 0; j < number; ++j) {
            pf.CallPacked(args, &temp);
          }
          t->Stop();
          int64_t t_nanos = t->SyncAndGetElapsedNanos();
          if (t_nanos == 0) absolute_zero_times++;
          duration_ms = t_nanos / 1e6;
        } 
      } while ((duration_ms < min_repeat_ms && absolute_zero_times < limit_zero_time_iterations) || (kyunam_invalid));

      // Generate the final value to be written to ostream (returned to python)
      double final_value;
      double delay = duration_ms / 1e3 / number; // unit: sec
      double power;
      if (static_cast<int>(dev.device_type) == static_cast<int>(kDLCPU)) power = energy_mJ / duration_ms; // unit: W
      else if (static_cast<int>(dev.device_type) == static_cast<int>(kDLCUDA)) power = power_W; 
      else power = 0;

      // Check if power capping is enabled (power cap set to >0)
      // If enabled, and measured power > cap, force the latency to be a large value
      const char* val = std::getenv("TVMP_POWER_CAP");
      int power_cap = (val != nullptr) ? std::atoi(val) : 0;
      if ((power_cap > 0) && (power > power_cap)) delay = delay + 50;

      // Final value contains both power and delay information
      power = std::floor(power * 10.0) / 10.0; // Truncate to 1 decimal place, so it doesn't interfere with latency part
      final_value = power * 1e3 + delay; // Assume delay < 100s
      std::cout << std::fixed << std::setprecision(9);
      std::cout << "<profiling.cc> power (W) = " << power << ", latency (us) = " << (delay * 1e6) << ", final_value = " << final_value << std::endl;

      // Check if the final_value is valid
      // if (delay >= 100) std::cout << "***** Latency >= 100 secs! final_value is corrupted. *****" << std::endl; 
      double recovered_power = std::floor((final_value / 1000.0) * 10.0) / 10.0;
      double recovered_delay = final_value - (recovered_power * 1000.0);
      if (std::fabs(recovered_power - power) / std::fabs(power) > 0.05) std::cout << "***** final_value is corrupted: recovered power is " << recovered_power << "W *****" << std::endl;
      if (std::fabs(recovered_delay - delay) / std::fabs(delay) > 0.05) std::cout << "***** final_value is corrupted: recovered delay is " << recovered_delay << "s *****" << std::endl;

      // Write the final value to ostream
      os.write(reinterpret_cast<char*>(&final_value), sizeof(final_value));

      if (cooldown_interval_ms > 0 && (i % repeats_to_cooldown) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_interval_ms));
      }
    }

    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  
  return PackedFunc(ftimer);
}

TVM_REGISTER_GLOBAL("runtime.profiling.Report")
    .set_body_typed([](Array<Map<String, ObjectRef>> calls,
                       Map<String, Map<String, ObjectRef>> device_metrics,
                       Map<String, ObjectRef> configuration) {
      return Report(calls, device_metrics, configuration);
    });

TVM_REGISTER_GLOBAL("runtime.profiling.Count").set_body_typed([](int64_t count) {
  return ObjectRef(make_object<CountNode>(count));
});

TVM_REGISTER_GLOBAL("runtime.profiling.Percent").set_body_typed([](double percent) {
  return ObjectRef(make_object<PercentNode>(percent));
});

TVM_REGISTER_GLOBAL("runtime.profiling.Duration").set_body_typed([](double duration) {
  return ObjectRef(make_object<DurationNode>(duration));
});

TVM_REGISTER_GLOBAL("runtime.profiling.Ratio").set_body_typed([](double ratio) {
  return ObjectRef(make_object<RatioNode>(ratio));
});

}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
