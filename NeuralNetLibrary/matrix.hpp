#include<initializer_list>
#include<vector>
#include<iostream>

template<typename T>
class matrix{
private:
    //Contains the values in the matrix
    std::vector<T> data;
    
    //Contains the dimensions of the matrix
    std::vector<int> dimensions;

    //Translates the matrix location to a vector index
    uint64_t elementNumber(std::initializer_list<uint64_t> location){
        //This is an error the location needs all the dimensions
        if(location.size() != dimensions.size()){
            std::invalid_argument("To access an element in a matrix all the dimensions need to be specified");
        }
        
        uint64_t index = 0;

        //Calculate the index
        for(int i = 0; i <  dimensions.size(); i++){
            //index += dimensions.at(i) + location.
        }
        
        return(index);
    }

public:
    //Returns the proper value given the matrix input
    T at(std::initializer_list<uint64_t> location){
        T value;

        value = data.at(elementNumber(location));
    }

    //pass in the the dimensions of the matrix
    matrix(std::initializer_list<uint64_t> list){
        uint64_t size = 1;

        for(auto dim : list){
            dimensions.push_back(dim);
            size *= dim;
        }

        data.resize(size);
    }

    ~matrix(){};
};
