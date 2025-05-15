#include "split_text.hpp"

#include <deque>
#include <ranges>

#include "string_util.hpp"

namespace ailoy {
#ifdef _WIN32
using size_t = unsigned long long;
#endif
using length_function_t = std::function<size_t(const std::string)>;

std::unordered_map<std::string, length_function_t> length_functions = {
    {"default", [](std::string s) { return s.size(); }},
    {"string", [](std::string s) { return s.size(); }},
};

void _check_chunk_overlap(const size_t chunk_size, const size_t chunk_overlap) {
  if (chunk_overlap > chunk_size)
    throw ailoy::exception(
        std::format("chunk_overlap({}) should be smaller than chunk size({}).",
                    chunk_overlap, chunk_size));
}

size_t calculate_total_direct(std::deque<std::string> current,
                              size_t separator_len, size_t new_len = 0) {
  size_t res = 0;
  if (current.empty())
    return new_len;
  for (auto e : current)
    res += e.size();
  res += (current.size() - 1) * separator_len;
  res += new_len ? separator_len + new_len : 0;
  return res;
}

std::vector<std::string> _merge_splits(const std::vector<std::string> splits,
                                       const std::string &separator,
                                       const size_t chunk_size,
                                       const size_t chunk_overlap,
                                       length_function_t flength) {
  auto separator_len = flength(separator);
  std::vector<std::string> docs;
  std::deque<std::string> current;

  auto complete_document = [&docs, separator](auto &current) {
    std::string doc = utils::join(separator, current);
    if (utils::trim(doc) != "")
      docs.push_back(doc);
  };

  size_t total = 0;
  for (auto s : splits) {
    auto new_len = flength(s);
    size_t total_if_merge =
        total + new_len + ((!current.empty()) ? separator_len : 0);
    if (total_if_merge > chunk_size) {
      // if(total > chunk_size) // warn?
      if (!current.empty()) {
        complete_document(current);
        /**
         * after completing document, pop from the current candidates until
         *  - overlap part length becomes small enough
         *  - room for the new candidate becomes large enough
         */
        while (total > chunk_overlap or
               (total_if_merge > chunk_size and total > 0)) {
          auto first = current.at(0);
          current.pop_front();
          total -= flength(first) + ((!current.empty()) ? separator_len : 0);
          total_if_merge =
              total + new_len + ((!current.empty()) ? separator_len : 0);
        }
      }
    }
    current.push_back(s);
    total = total_if_merge;
  }

  complete_document(current);

  return docs;
}

std::vector<std::string>
split_text_by_separator(const std::string text, const size_t chunk_size,
                        const size_t chunk_overlap, const std::string separator,
                        const std::string length_function) {
  _check_chunk_overlap(chunk_size, chunk_overlap);
  length_function_t flength = length_functions.at(length_function);
  size_t len_sep = flength(separator);
  auto splits = utils::split_text(text, separator);
  return _merge_splits(splits, separator, chunk_size, chunk_overlap, flength);
}

const std::string
_pick_best_separator(const std::string &text,
                     const std::vector<std::string> &separators,
                     std::vector<std::string> &new_separators) {
  std::string rv_sep = separators.back();
  bool push = false;
  for (size_t i = 0; i < separators.size(); i++) {
    auto sep = separators.at(i);
    // push the remaining separators after setting rv_sep
    if (push) {
      new_separators.push_back(sep);
      continue;
    }

    if (sep == "") {
      // treat empty separator especially
      rv_sep = sep;
      break;
    } else if (text.find(sep) != std::string::npos) {
      // pick the next sep if the text contains it
      rv_sep = sep;
      push = true;
    }
  }
  return rv_sep;
}

void _merge_splits_and_put(std::vector<std::string> &chunks,
                           const std::vector<std::string> &splits,
                           const std::string &separator,
                           const size_t chunk_size, const size_t chunk_overlap,
                           length_function_t flength) {
  auto merged_chunks =
      _merge_splits(splits, separator, chunk_size, chunk_overlap, flength);
  for (auto chunk : merged_chunks)
    chunks.push_back(chunk);
}

std::vector<std::string> _split_text_recursive(
    const std::string text, const size_t chunk_size, const size_t chunk_overlap,
    const std::vector<std::string> &separators, length_function_t flength) {
  std::vector<std::string> new_separators{};
  const std::string separator =
      _pick_best_separator(text, separators, new_separators);
  auto splits = utils::split_text(text, separator);

  std::vector<std::string> _good_splits{};
  std::vector<std::string> final_chunks{};
  for (const std::string s : splits) {
    if (flength(s) < chunk_size) // gather 'good splits'
      _good_splits.push_back(s);
    else { // if a split it too big
      // first, merge good splits into chunks and put them into final chunks.
      // merges are only on the splits in the same step.
      if (!_good_splits.empty()) {
        _merge_splits_and_put(final_chunks, _good_splits, separator, chunk_size,
                              chunk_overlap, flength);
        _good_splits.clear();
      }

      // split up the big split if there are more steps
      if (new_separators.empty())
        final_chunks.push_back(s);
      else {
        // put the chunks from next steps into final chunks.
        auto chunks_from_next_steps = _split_text_recursive(
            s, chunk_size, chunk_overlap, new_separators, flength);
        for (auto chunk : chunks_from_next_steps)
          final_chunks.push_back(chunk);
      }
    }
  }
  // process remaining good splits
  if (!_good_splits.empty()) {
    _merge_splits_and_put(final_chunks, _good_splits, separator, chunk_size,
                          chunk_overlap, flength);
  }
  return final_chunks;
}

std::vector<std::string> split_text_by_separators_recursively(
    const std::string text, const size_t chunk_size, const size_t chunk_overlap,
    const std::vector<std::string> separators,
    const std::string length_function) {
  _check_chunk_overlap(chunk_size, chunk_overlap);
  length_function_t flength = length_functions.at(length_function);
  return _split_text_recursive(text, chunk_size, chunk_overlap, separators,
                               flength);
}

value_or_error_t
split_text_by_separator_op(std::shared_ptr<const value_t> inputs) {
  // Get input parameters
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("Split Text", "inputs", "map_t", inputs->get_type()));
  auto input_map = inputs->as<map_t>();

  // Parse text
  if (!input_map->contains("text"))
    return error_output_t(range_error("Split Text", "text"));
  if (!input_map->at("text")->is_type_of<string_t>())
    return error_output_t(type_error("Split Text", "text", "string_t",
                                     input_map->at("text")->get_type()));
  const std::string &text = *input_map->at<string_t>("text");

  // Parse chunk size
  size_t chunk_size = 4000;
  if (input_map->contains("chunk_size")) {
    if (input_map->at("chunk_size")->is_type_of<uint_t>())
      chunk_size = *input_map->at<uint_t>("chunk_size");
    else if (input_map->at("chunk_size")->is_type_of<int_t>())
      chunk_size = *input_map->at<int_t>("chunk_size");
    else
      return error_output_t(
          type_error("Split Text", "chunk_size", "uint_t | int_t",
                     input_map->at("chunk_size")->get_type()));
    if (chunk_size < 1)
      return error_output_t(value_error("Split Text", "chunk_size", ">= 1",
                                        std::to_string(chunk_size)));
  }

  // Parse chunk overlap
  size_t chunk_overlap = 200;
  if (input_map->contains("chunk_overlap")) {
    if (input_map->at("chunk_overlap")->is_type_of<uint_t>())
      chunk_overlap = *input_map->at<uint_t>("chunk_overlap");
    else if (input_map->at("chunk_overlap")->is_type_of<int_t>())
      chunk_overlap = *input_map->at<int_t>("chunk_overlap");
    else
      return error_output_t(
          type_error("Split Text", "chunk_overlap", "uint_t | int_t",
                     input_map->at("chunk_overlap")->get_type()));
    if (chunk_overlap < 1)
      return error_output_t(value_error("Split Text", "chunk_overlap", ">= 1",
                                        std::to_string(chunk_overlap)));
  }

  // Parse separator
  std::string separator = "\n\n";
  if (input_map->contains("separator")) {
    if (!input_map->at("separator")->is_type_of<string_t>())
      return error_output_t(type_error("Split Text", "separator", "string_t",
                                       input_map->at("separator")->get_type()));
    separator = *input_map->at<string_t>("separator");
  }

  // Parse length_function
  std::string length_function = "default";
  if (input_map->contains("length_function")) {
    if (!input_map->at("length_function")->is_type_of<string_t>())
      return error_output_t(
          type_error("Split Text", "length_function", "string_t",
                     input_map->at("length_function")->get_type()));
    length_function = *input_map->at<string_t>("length_function");
    if (!(length_function == "default" || length_function == "string"))
      return error_output_t(value_error("Split Text", "length_function",
                                        "default | string", length_function));
  }

  // Split text into chunks
  auto chunks = split_text_by_separator(text, chunk_size, chunk_overlap,
                                        separator, length_function);

  // Return output
  auto outputs = create<map_t>();
  outputs->insert_or_assign("chunks", create<array_t>());
  for (const auto &chunk : chunks)
    outputs->at<array_t>("chunks")->push_back(create<string_t>(chunk));
  return outputs;
}

value_or_error_t
split_text_by_separators_recursively_op(std::shared_ptr<const value_t> inputs) {
  // Get input parameters
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("Split Text", "inputs", "map_t", inputs->get_type()));
  auto input_map = inputs->as<map_t>();

  // Parse text
  if (!input_map->contains("text"))
    return error_output_t(range_error("Split Text", "text"));
  if (!input_map->at("text")->is_type_of<string_t>())
    return error_output_t(type_error("Split Text", "text", "string_t",
                                     input_map->at("text")->get_type()));
  const std::string &text = *input_map->at<string_t>("text");

  // Parse chunk size
  size_t chunk_size = 4000;
  if (input_map->contains("chunk_size")) {
    if (input_map->at("chunk_size")->is_type_of<uint_t>())
      chunk_size = *input_map->at<uint_t>("chunk_size");
    else if (input_map->at("chunk_size")->is_type_of<int_t>())
      chunk_size = *input_map->at<int_t>("chunk_size");
    else
      return error_output_t(
          type_error("Split Text", "chunk_size", "uint_t | int_t",
                     input_map->at("chunk_size")->get_type()));
    if (chunk_size < 1)
      return error_output_t(value_error("Split Text", "chunk_size", ">= 1",
                                        std::to_string(chunk_size)));
  }

  // Parse chunk overlap
  size_t chunk_overlap = 200;
  if (input_map->contains("chunk_overlap")) {
    if (input_map->at("chunk_overlap")->is_type_of<uint_t>())
      chunk_overlap = *input_map->at<uint_t>("chunk_overlap");
    else if (input_map->at("chunk_overlap")->is_type_of<int_t>())
      chunk_overlap = *input_map->at<int_t>("chunk_overlap");
    else
      return error_output_t(
          type_error("Split Text", "chunk_overlap", "uint_t | int_t",
                     input_map->at("chunk_overlap")->get_type()));
    if (chunk_overlap < 1)
      return error_output_t(value_error("Split Text", "chunk_overlap", ">= 1",
                                        std::to_string(chunk_overlap)));
  }

  // Parse separators
  std::vector<std::string> separators;
  if (input_map->contains("separator")) {
    if (!input_map->at("separators")->is_type_of<array_t>())
      return error_output_t(
          type_error("Split Text", "separators", "array_t",
                     input_map->at("separators")->get_type()));
    for (const auto sep : *input_map->at<array_t>("separators")) {
      if (!sep->is_type_of<string_t>())
        return error_output_t(type_error("Split Text", "separators.*",
                                         "string_t", sep->get_type()));
      separators.push_back(*sep->as<string_t>());
    }
  } else {
    separators.push_back("\n\n");
    separators.push_back("\n");
    separators.push_back(" ");
    separators.push_back("");
  }

  // Parse length_function
  std::string length_function = "default";
  if (input_map->contains("length_function")) {
    if (!input_map->at("length_function")->is_type_of<string_t>())
      return error_output_t(
          type_error("Split Text", "length_function", "string_t",
                     input_map->at("length_function")->get_type()));
    length_function = *input_map->at<string_t>("length_function");
    if (!(length_function == "default" || length_function == "string"))
      return error_output_t(value_error("Split Text", "length_function",
                                        "default | string", length_function));
  }

  // Split text into chunks
  auto chunks = split_text_by_separators_recursively(
      text, chunk_size, chunk_overlap, separators, length_function);

  // Return output
  auto outputs = create<map_t>();
  outputs->insert_or_assign("chunks", create<array_t>());
  for (const auto &chunk : chunks)
    outputs->at<array_t>("chunks")->push_back(create<string_t>(chunk));
  return outputs;
}

} // namespace ailoy
