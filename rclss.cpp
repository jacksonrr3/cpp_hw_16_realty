
//#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>
//#include <dlib/clustering.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>

using namespace dlib;

using sample_type = matrix<double, 7, 1>;
using linear_kernel_type = linear_kernel<sample_type>;
using ovo_trainer_type = one_vs_one_trainer<any_trainer<sample_type>>;
using ovo_df_type = one_vs_one_decision_function<ovo_trainer_type, decision_function<linear_kernel_type>>;

int main(int argc, char* argv[])
{
    try
    {
        std::map<std::size_t, std::vector<sample_type>> samples;

     //   int clusters;
        std::string modelfname;

        if (argc == 2)
        {
            modelfname = argv[1];
        }
        else {
            std::cerr << "rclss <modelfname>\n";
            return 1;
        }

        std::ifstream in_file;
        std::string input_string;
        std::string token;
        sample_type m;
 

        in_file.open(modelfname + ".csv");
        if (!in_file) {
            std::cout << "Error read model fail.\n";
            return 1;
        }
      //  in_file >> clusters;
      //  std::getline(in_file, input_string);

        //read data from file
        while (std::getline(in_file, input_string)) {
            std::stringstream ss(input_string);
            for (int i = 0; i < 8; i++) {
                std::getline(ss, token, ';');
                if (i != 7) {
                    m(i) = std::stod(token);
                }
                else {
                    samples[std::stod(token)].push_back(m);
                }
            }
        }

        ovo_df_type df;
        deserialize(modelfname + ".df") >> df;

        std::string request;
        while(std::getline(std::cin, request)) {
            std::stringstream ss_req(request);
           // std::stringstream ss_req("86.116781;55.335492;2;4326901.00;54.00;7.00;1\n");
            for (int i = 0; i < 7; i++) {
                std::getline(ss_req, token, ';');
                std::cout << i << std::endl;
                m(i) = std::stod(token);
            }

            auto label = df(m);

            std::sort(samples[label].begin(), samples[label].end(),
                [&m](const auto& a, const auto& b) {
                    double r1 = std::pow((a(0) - m(0)), 2) + std::pow((a(1) - m(1)), 2);
                    double r2 = std::pow((b(0) - m(0)), 2) + std::pow((b(1) - m(1)), 2);
                    return r1 < r2;
                }
                );

            std::cout << "Result objects:" << std::endl;
            for (auto& s : samples[label]) {
                std::cout << std::to_string(s(0)) << ";" <<
                    std::to_string(s(1)) << ";" <<
                    std::to_string(s(2)) << ";" <<
                    std::to_string(s(3)) << ";" <<
                    std::to_string(s(4)) << ";" <<
                    std::to_string(s(5)) << ";" <<
                    std::to_string(s(6)) << std::endl;
            }

        }

    }
    catch (std::exception & e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
