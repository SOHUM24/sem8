#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>

using namespace std;

void sequentialbubble(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelbubble(int arr[], int n)
{
#pragma omp parallel for shared(arr)
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void generaterandomarray(int arr[], int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 10000;
    }
}

void displayRandomArray(int arr[], int n)
{
    cout << "Randomly generated array:" << endl;
    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
}


int main()
{
    int size;
    cout << "enter array size: ";
    cin >> size;
    int *arr = new int[size]; // Dynamic memory allocation

    generaterandomarray(arr, size);
    displayRandomArray(arr, size);

    clock_t startTime = clock();
    sequentialbubble(arr, size);
    clock_t endTime = clock();
    cout << "performance:" << double(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << endl;

    generaterandomarray(arr, size);

    startTime = clock();
    parallelbubble(arr, size);
    endTime = clock();
    cout << "performance:" << double(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << endl;

    delete[] arr; // Free the dynamically allocated memory

    return 0;
}
