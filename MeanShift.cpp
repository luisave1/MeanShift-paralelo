#include "MeanShift.h" // Se asume que este archivo de encabezado declara la clase MeanShift y la estructura Point5D
#include <cmath>
#include <cstring>
#include <omp.h> // OpenMP para procesamiento paralelo

#define MS_MAX_NUM_CONVERGENCE_STEPS 5   // Máximo número de iteraciones para la convergencia de Mean Shift
#define MS_MEAN_SHIFT_TOL_COLOR 0.3     // Tolerancia para la distancia de color en Mean Shift
#define MS_MEAN_SHIFT_TOL_SPATIAL 0.3   // Tolerancia para la distancia espacial en Mean Shift

// Array para representar los offsets de los 8 vecinos de un píxel
const int dxdy[][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };

// Clase para representar un punto 5D (x, y, l, a, b)
Point5D::Point5D() : x(-1), y(-1), l(0), a(0), b(0) {}
Point5D::~Point5D() {}

// Convierte RGB al espacio de color Lab
void Point5D::PointLab() {
    l = l * 100 / 255;  // Normaliza L a [0, 100]
    a = a - 128;        // Desplaza a para centrarlo alrededor de 0
    b = b - 128;        // Desplaza b para centrarlo alrededor de 0
}

// Convierte Lab al espacio de color RGB. Nota: Esta conversión es aproximada y para visualización.
void Point5D::PointRGB() {
    l = l * 255 / 100;  // Escala L de vuelta a [0, 255]
    a = a + 128;        // Desplaza a de vuelta
    b = b + 128;        // Desplaza b de vuelta
}

// Acumula los componentes de otro Point5D
void Point5D::MSPoint5DAccum(Point5D Pt) {
    x += Pt.x; y += Pt.y; l += Pt.l; a += Pt.a; b += Pt.b;
}

// Copia los componentes de otro Point5D
void Point5D::MSPoint5DCopy(Point5D Pt) {
    x = Pt.x; y = Pt.y; l = Pt.l; a = Pt.a; b = Pt.b;
}

// Calcula la distancia de color entre dos Point5D en el espacio Lab
float Point5D::MSPoint5DColorDistance(Point5D Pt) {
    return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

// Calcula la distancia espacial entre dos Point5D
float Point5D::MSPoint5DSpatialDistance(Point5D Pt) {
    return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

// Escala los componentes del Point5D
void Point5D::MSPoint5DScale(float scale) {
    x *= scale; y *= scale; l *= scale; a *= scale; b *= scale;
}

// Establece los componentes del Point5D
void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb) {
    x = px; y = py; l = pl; a = pa; b = pb;
}

// Imprime los componentes del Point5D en la consola
void Point5D::Print() {
    cout << x << " " << y << " " << l << " " << a << " " << b << endl;
}

// Constructor para la clase MeanShift
MeanShift::MeanShift(float s, float r) : hs(s), hr(r) {}
// hs: ancho de banda espacial, hr: ancho de banda de color

// Realiza el filtrado Mean Shift en la imagen de entrada
void MeanShift::MSFiltering(Mat& Img) {
    int ROWS = Img.rows, COLS = Img.cols; // Obtiene las dimensiones de la imagen
    split(Img, IMGChannels);             // Divide la imagen en sus canales de color (asumiendo el orden BGR en OpenCV)

    // Procesamiento paralelo usando OpenMP. Los bucles externo e interno están paralelizados.
#pragma omp parallel for collapse(2)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            Point5D PtCur, PtPrev, PtSum, Pt; // Declara variables Point5D
            // Define los límites de la ventana de búsqueda
            int Left = max(0, j - (int)hs), Right = min(COLS, j + (int)hs);
            int Top = max(0, i - (int)hs), Bottom = min(ROWS, i + (int)hs);

            // Inicializa el punto actual con su información espacial y de color
            PtCur.MSPOint5DSet(i, j,
                (float)IMGChannels[0].at<uchar>(i, j), // Canal azul
                (float)IMGChannels[1].at<uchar>(i, j), // Canal verde
                (float)IMGChannels[2].at<uchar>(i, j)); // Canal rojo
            PtCur.PointLab(); // Convierte el color al espacio de color Lab

            int step = 0, NumPts;
            // Bucle de iteración de Mean Shift
            do {
                PtPrev.MSPoint5DCopy(PtCur); // Almacena el punto anterior
                PtSum.MSPOint5DSet(0, 0, 0, 0, 0); // Resetea la suma
                NumPts = 0; // Resetea el número de puntos en el kernel
                // Itera sobre la ventana de búsqueda
                for (int hx = Top; hx < Bottom; hx++) {
                    for (int hy = Left; hy < Right; hy++) {
                        Pt.MSPOint5DSet(hx, hy,
                            (float)IMGChannels[0].at<uchar>(hx, hy),
                            (float)IMGChannels[1].at<uchar>(hx, hy),
                            (float)IMGChannels[2].at<uchar>(hx, hy));
                        Pt.PointLab(); // Convierte a Lab
                        // Si la distancia de color está dentro del ancho de banda, agrega el punto a la suma
                        if (Pt.MSPoint5DColorDistance(PtCur) < hr) {
                            PtSum.MSPoint5DAccum(Pt);
                            NumPts++;
                        }
                    }
                }
                PtSum.MSPoint5DScale(1.0f / NumPts); // Calcula la media
                PtCur.MSPoint5DCopy(PtSum);         // Actualiza el punto actual
                step++;
                // Comprueba la convergencia: distancia de color, distancia espacial y número máximo de iteraciones
            } while (PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR &&
                PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL &&
                step < MS_MAX_NUM_CONVERGENCE_STEPS);

            PtCur.PointRGB(); // Convierte el punto final de vuelta a RGB
            Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b); // Actualiza la imagen con el color filtrado
        }
    }
}

// Realiza la segmentación Mean Shift en la imagen de entrada
void MeanShift::MSSegmentation(Mat& Img) {
    MSFiltering(Img); // Primero, aplica el filtrado Mean Shift para suavizar la imagen
    int ROWS = Img.rows, COLS = Img.cols;
    split(Img, IMGChannels);

    int label = -1; // Inicializa la etiqueta del segmento
    float* Mode = new float[ROWS * COLS * 3];       // Array para almacenar el modo (pico) del color de cada segmento
    int* MemberModeCount = new int[ROWS * COLS]; // Array para almacenar el número de píxeles en cada segmento
    memset(MemberModeCount, 0, ROWS * COLS * sizeof(int)); // Inicializa los conteos a 0

    int** Labels = new int* [ROWS]; // Array 2D para almacenar la etiqueta del segmento para cada píxel
    for (int i = 0; i < ROWS; i++)
        Labels[i] = new int[COLS];

// Paraleliza la inicialización de las etiquetas
#pragma omp parallel for collapse(2)
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            Labels[i][j] = -1; // Inicializa todas las etiquetas a -1 (sin etiquetar)

    // Itera a través de cada píxel en la imagen
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (Labels[i][j] < 0) { // Si el píxel aún no está etiquetado
                Point5D PtCur, Pt, P;
                label++;             // Incrementa la etiqueta del segmento
                Labels[i][j] = label; // Asigna la etiqueta actual al píxel
                PtCur.MSPOint5DSet(i, j,  // Inicializa el punto actual
                    (float)IMGChannels[0].at<uchar>(i, j),
                    (float)IMGChannels[1].at<uchar>(i, j),
                    (float)IMGChannels[2].at<uchar>(i, j));
                PtCur.PointLab();

                // Almacena el modo inicial (color) del segmento
                Mode[label * 3 + 0] = PtCur.l;
                Mode[label * 3 + 1] = PtCur.a;
                Mode[label * 3 + 2] = PtCur.b;

                vector<Point5D> NeighbourPoints; // Vector para almacenar los píxeles vecinos
                NeighbourPoints.push_back(PtCur);  // Agrega el píxel actual a los vecinos

                // Crecimiento de la región: encuentra los píxeles conectados que pertenecen al mismo segmento
                while (!NeighbourPoints.empty()) {
                    Pt = NeighbourPoints.back();
                    NeighbourPoints.pop_back();

                    // Itera sobre los 8 vecinos del píxel actual
                    for (int k = 0; k < 8; k++) {
                        int hx = Pt.x + dxdy[k][0], hy = Pt.y + dxdy[k][1];
                        // Comprueba si el vecino está dentro de los límites de la imagen y sin etiquetar
                        if (hx >= 0 && hy >= 0 && hx < ROWS && hy < COLS && Labels[hx][hy] < 0) {
                            P.MSPOint5DSet(hx, hy, // Inicializa el punto vecino
                                (float)IMGChannels[0].at<uchar>(hx, hy),
                                (float)IMGChannels[1].at<uchar>(hx, hy),
                                (float)IMGChannels[2].at<uchar>(hx, hy));
                            P.PointLab();
                            // Si el color del vecino está cerca del color del píxel actual
                            if (PtCur.MSPoint5DColorDistance(P) < hr) {
                                Labels[hx][hy] = label;      // Asigna la misma etiqueta
                                NeighbourPoints.push_back(P); // Agrega el vecino a la cola
                                MemberModeCount[label]++;     // Incrementa el tamaño del segmento
                                // Acumula el color para el cálculo del modo
                                Mode[label * 3 + 0] += P.l;
                                Mode[label * 3 + 1] += P.a;
                                Mode[label * 3 + 2] += P.b;
                            }
                        }
                    }
                }
                MemberModeCount[label]++; // Incrementa el tamaño del segmento.
                // Calcula el color promedio (modo) del segmento
                Mode[label * 3 + 0] /= MemberModeCount[label];
                Mode[label * 3 + 1] /= MemberModeCount[label];
                Mode[label * 3 + 2] /= MemberModeCount[label];
            }
        }
    }

    // Asigna el color del modo a cada píxel en la imagen basado en su etiqueta
#pragma omp parallel for collapse(2)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int lbl = Labels[i][j]; // Obtiene la etiqueta del píxel actual
            Point5D Pixel;
            Pixel.MSPOint5DSet(i, j,
                Mode[lbl * 3 + 0],
                Mode[lbl * 3 + 1],
                Mode[lbl * 3 + 2]); // Establece el color del píxel al color del modo de su segmento.
            Pixel.PointRGB();
            Img.at<Vec3b>(i, j) = Vec3b(Pixel.l, Pixel.a, Pixel.b); // Actualiza la imagen.
        }
    }

    // Limpia la memoria asignada
    delete[] Mode;
    delete[] MemberModeCount;
    for (int i = 0; i < ROWS; i++) delete[] Labels[i];
    delete[] Labels;
}
