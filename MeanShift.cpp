#include "MeanShift.h"
#include <cmath>
#include <cstring>
#include <omp.h> // OpenMP

#define MS_MAX_NUM_CONVERGENCE_STEPS 5
#define MS_MEAN_SHIFT_TOL_COLOR 0.3
#define MS_MEAN_SHIFT_TOL_SPATIAL 0.3
const int dxdy[][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };

Point5D::Point5D() : x(-1), y(-1), l(0), a(0), b(0) {}
Point5D::~Point5D() {}

void Point5D::PointLab() {
	l = l * 100 / 255;
	a = a - 128;
	b = b - 128;
}
void Point5D::PointRGB() {
	l = l * 255 / 100;
	a = a + 128;
	b = b + 128;
}
void Point5D::MSPoint5DAccum(Point5D Pt) {
	x += Pt.x; y += Pt.y; l += Pt.l; a += Pt.a; b += Pt.b;
}
void Point5D::MSPoint5DCopy(Point5D Pt) {
	x = Pt.x; y = Pt.y; l = Pt.l; a = Pt.a; b = Pt.b;
}
float Point5D::MSPoint5DColorDistance(Point5D Pt) {
	return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}
float Point5D::MSPoint5DSpatialDistance(Point5D Pt) {
	return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}
void Point5D::MSPoint5DScale(float scale) {
	x *= scale; y *= scale; l *= scale; a *= scale; b *= scale;
}
void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb) {
	x = px; y = py; l = pl; a = pa; b = pb;
}
void Point5D::Print() {
	cout << x << " " << y << " " << l << " " << a << " " << b << endl;
}

MeanShift::MeanShift(float s, float r) : hs(s), hr(r) {}

void MeanShift::MSFiltering(Mat& Img) {
	int ROWS = Img.rows, COLS = Img.cols;
	split(Img, IMGChannels);

#pragma omp parallel for collapse(2)
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			Point5D PtCur, PtPrev, PtSum, Pt;
			int Left = max(0, j - (int)hs), Right = min(COLS, j + (int)hs);
			int Top = max(0, i - (int)hs), Bottom = min(ROWS, i + (int)hs);

			PtCur.MSPOint5DSet(i, j,
				(float)IMGChannels[0].at<uchar>(i, j),
				(float)IMGChannels[1].at<uchar>(i, j),
				(float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();

			int step = 0, NumPts;
			do {
				PtPrev.MSPoint5DCopy(PtCur);
				PtSum.MSPOint5DSet(0, 0, 0, 0, 0);
				NumPts = 0;
				for (int hx = Top; hx < Bottom; hx++) {
					for (int hy = Left; hy < Right; hy++) {
						Pt.MSPOint5DSet(hx, hy,
							(float)IMGChannels[0].at<uchar>(hx, hy),
							(float)IMGChannels[1].at<uchar>(hx, hy),
							(float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();
						if (Pt.MSPoint5DColorDistance(PtCur) < hr) {
							PtSum.MSPoint5DAccum(Pt);
							NumPts++;
						}
					}
				}
				PtSum.MSPoint5DScale(1.0f / NumPts);
				PtCur.MSPoint5DCopy(PtSum);
				step++;
			} while (PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR &&
				PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL &&
				step < MS_MAX_NUM_CONVERGENCE_STEPS);

			PtCur.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}
}

void MeanShift::MSSegmentation(Mat& Img) {
	MSFiltering(Img); // Reutiliza el código anterior
	int ROWS = Img.rows, COLS = Img.cols;
	split(Img, IMGChannels);

	int label = -1;
	float* Mode = new float[ROWS * COLS * 3];
	int* MemberModeCount = new int[ROWS * COLS];
	memset(MemberModeCount, 0, ROWS * COLS * sizeof(int));

	int** Labels = new int* [ROWS];
	for (int i = 0; i < ROWS; i++)
		Labels[i] = new int[COLS];

#pragma omp parallel for collapse(2)
	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLS; j++)
			Labels[i][j] = -1;

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			if (Labels[i][j] < 0) {
				Point5D PtCur, Pt, P;
				label++;
				Labels[i][j] = label;
				PtCur.MSPOint5DSet(i, j,
					(float)IMGChannels[0].at<uchar>(i, j),
					(float)IMGChannels[1].at<uchar>(i, j),
					(float)IMGChannels[2].at<uchar>(i, j));
				PtCur.PointLab();

				Mode[label * 3 + 0] = PtCur.l;
				Mode[label * 3 + 1] = PtCur.a;
				Mode[label * 3 + 2] = PtCur.b;

				vector<Point5D> NeighbourPoints;
				NeighbourPoints.push_back(PtCur);
				while (!NeighbourPoints.empty()) {
					Pt = NeighbourPoints.back();
					NeighbourPoints.pop_back();

					for (int k = 0; k < 8; k++) {
						int hx = Pt.x + dxdy[k][0], hy = Pt.y + dxdy[k][1];
						if (hx >= 0 && hy >= 0 && hx < ROWS && hy < COLS && Labels[hx][hy] < 0) {
							P.MSPOint5DSet(hx, hy,
								(float)IMGChannels[0].at<uchar>(hx, hy),
								(float)IMGChannels[1].at<uchar>(hx, hy),
								(float)IMGChannels[2].at<uchar>(hx, hy));
							P.PointLab();
							if (PtCur.MSPoint5DColorDistance(P) < hr) {
								Labels[hx][hy] = label;
								NeighbourPoints.push_back(P);
								MemberModeCount[label]++;
								Mode[label * 3 + 0] += P.l;
								Mode[label * 3 + 1] += P.a;
								Mode[label * 3 + 2] += P.b;
							}
						}
					}
				}
				MemberModeCount[label]++;
				Mode[label * 3 + 0] /= MemberModeCount[label];
				Mode[label * 3 + 1] /= MemberModeCount[label];
				Mode[label * 3 + 2] /= MemberModeCount[label];
			}
		}
	}

#pragma omp parallel for collapse(2)
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			int lbl = Labels[i][j];
			Point5D Pixel;
			Pixel.MSPOint5DSet(i, j,
				Mode[lbl * 3 + 0],
				Mode[lbl * 3 + 1],
				Mode[lbl * 3 + 2]);
			Pixel.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(Pixel.l, Pixel.a, Pixel.b);
		}
	}

	delete[] Mode;
	delete[] MemberModeCount;
	for (int i = 0; i < ROWS; i++) delete[] Labels[i];
	delete[] Labels;
}