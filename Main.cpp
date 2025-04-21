//Main.cpp
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp> // Incluye las funcionalidades de OpenCV
#include "MeanShift.h"       // Incluye la definición de la clase MeanShift

using namespace cv;
using namespace std;

int main() {
    // Cargar la imagen
    Mat Img = imread("C:\\Users\\LUIS FERNANDO\\Pictures\\arte\\THL.jpg");
    if (Img.empty()) {
        cerr << "Error: No se pudo abrir o encontrar la imagen 'THL.jpg'" << endl;
        return -1;
    }
    resize(Img, Img, Size(256, 256), 0, 0, INTER_LINEAR); // Redimensionar la imagen

    // Mostrar la imagen original
    namedWindow("The Original Picture");
    imshow("The Original Picture", Img);

    auto start = std::chrono::high_resolution_clock::now();
    // Convertir el espacio de color de RGB (BGR en OpenCV) a Lab
    cvtColor(Img, Img, COLOR_BGR2Lab);

    // Inicializar el objeto MeanShift con los anchos de banda espacial y de color
    MeanShift MSProc(8, 16); // hs = 8 (spatial), hr = 16 (color)

    // Procesamiento de filtrado Mean Shift
    MSProc.MSFiltering(Img);

    /*Procesamiento de segmentación Mean Shift (incluye el filtrado)*/
    MSProc.MSSegmentation(Img); // Descomentar para probar la segmentación

    // Imprimir los anchos de banda utilizados
    cout << "El ancho de banda espacial es " << MSProc.hs << endl;
    cout << "El ancho de banda de color es " << MSProc.hr << endl;

    // Convertir el espacio de color de Lab de vuelta a RGB (BGR en OpenCV)
    cvtColor(Img, Img, COLOR_Lab2BGR);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms" << std::endl;
    // Mostrar la imagen resultante del Mean Shift
    namedWindow("MS Picture");
    imshow("MS Picture", Img);

    waitKey(0); // Esperar a que se presione una tecla para cerrar las ventanas
    return 0;
}
