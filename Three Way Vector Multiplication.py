%%writefile lab/simple1.cpp
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

// Define the 3-way vector multiplication kernel
void vectorMultiplication(queue& q, buffer<float, 1>& input1, buffer<float, 1>& input2, buffer<float, 1>& input3, buffer<float, 1>& output) {
  q.submit([&](handler& h) {
    // Define accessor objects to read from the input buffers and write to the output buffer
    auto in1 = input1.get_access<access::mode::read>(h);
    auto in2 = input2.get_access<access::mode::read>(h);
    auto in3 = input3.get_access<access::mode::read>(h);
    auto out = output.get_access<access::mode::write>(h);

    // Define the range of indices to operate on
    range<1> r(output.get_count());

    // Define a nd_range object to specify the global and local work group sizes
    nd_range<1> ndr(r, range<1>(16), id<1>(0));


    // Define a kernel function to perform the vector multiplication
    h.parallel_for(ndr, [=](nd_item<1> item) {
      // Get the global index of the current work item
      int i = item.get_global_id(0);

      // Multiply the corresponding elements of the input vectors and store the result in the output vector
      out[i] = in1[i] * in2[i] * in3[i];
    });
  });
}

int main() {
  // Initialize the input data
  std::vector<float> input1Data = { 1, 2, 3, 4, 5 };
  std::vector<float> input2Data = { 2, 4, 6, 8, 10 };
  std::vector<float> input3Data = { 3, 6, 9, 12, 15 };

  // Create SYCL buffers to hold the input and output data
  buffer<float, 1> input1(input1Data.data(), range<1>(input1Data.size()));
  buffer<float, 1> input2(input2Data.data(), range<1>(input2Data.size()));
  buffer<float, 1> input3(input3Data.data(), range<1>(input3Data.size()));
  buffer<float, 1> output(input1Data.size());

  // Create a SYCL queue to execute the kernel on
  queue q;

  // Call the 3-way vector multiplication kernel
  vectorMultiplication(q, input1, input2, input3, output);

  // Transfer the results back to the host
  std::vector<float> outputData(input1Data.size());
  q.submit([&](handler& h) {
    auto out = output.get_access<access::mode::read>(h);
    h.copy(out, outputData.data());
  }).wait();

  // Display the output data
  for (auto f : outputData) {
    std::cout << f << " ";
  }

  return 0;
}
