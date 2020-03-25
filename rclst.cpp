
#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>
#include <dlib/clustering.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace dlib;
using sample_type = matrix<double, 7, 1>;

using linear_kernel_type = linear_kernel<sample_type>;
using ovo_trainer_type = one_vs_one_trainer<any_trainer<sample_type>>;
using ovo_df_type = one_vs_one_decision_function<ovo_trainer_type, decision_function<linear_kernel_type>>;

int main(int argc, char* argv[])
{
    try
    {
        std::vector<sample_type> samples;
        std::vector<sample_type> initial_centers;
        std::vector<double> labels;

        int clusters;
        std::string modelfname;

        if (argc == 3)
        {
            clusters = atoi(argv[1]);
            modelfname = argv[2];
            if (clusters < 3){
                std::cout << "need more clusters! \n";
                return 1;
            }
        }
        else{
            std::cerr << "rclst <clusters> <modelfname>\n";
            return 1;
        }

       // std::ifstream if_file;
       // if_file.open("int_test.csv");

        std::string input_string;
        sample_type m;
       // while (std::getline(if_file, input_string)) {
        while (std::getline(std::cin, input_string)) {
            std::stringstream ss(input_string);
            std::string token;
            for (int i = 0; i < 8; i++) {
                std::getline(ss, token, ';');
                if (token == "") {
                    token = "0.0";
                }

                if (i == 7) {
                    double d = std::stod(token);
                    if (d == m(i - 1) || m(i - 1) == 1 || d == 0 || m(i - 1) == 0) {
                        m(i - 1) = 0;
                    }
                    else {
                        m(i - 1) = 1;
                    }
                }
                else {
                    m(i) = std::stod(token);
                }
            }

            samples.push_back(m);
        }

        kcentroid<linear_kernel_type> kc(linear_kernel_type(), 0.01, 8);
        kkmeans<linear_kernel_type> test(kc);

        test.set_number_of_centers(clusters);
        pick_initial_centers(clusters, initial_centers, samples, test.get_kernel());
        find_clusters_using_kmeans(samples, initial_centers);
        test.train(samples, initial_centers);

        std::ofstream out_file(modelfname + ".csv");
       // out_file << clusters << "\n";
        for (auto& s : samples) {
            double l = test(s);
            out_file << std::to_string(s(0)) << ";" <<
                std::to_string(s(1)) << ";" <<
                std::to_string(s(2)) << ";" <<
                std::to_string(s(3)) << ";" <<
                std::to_string(s(4)) << ";" <<
                std::to_string(s(5)) << ";" <<
                std::to_string(s(6)) << ";" <<
                std::to_string(l) << "\n";
            labels.push_back(l);
        }

        ovo_trainer_type ovo_trainer;

        krr_trainer<linear_kernel_type> krr_lin_trainer;
        ovo_trainer.set_trainer(krr_lin_trainer);
        ovo_df_type df = ovo_trainer.train(samples, labels);

        serialize(modelfname + ".df") << df;

    }
    catch (std::exception & e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
    }
    return 0;
}
