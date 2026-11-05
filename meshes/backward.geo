
cl_v = 0.9;
Point(2) = {22, 5, -0, cl_v};
//+
Point(4) = {4, 2, -0, 0.3};
//+
Point(6) = {0, 2, -0, cl_v};
//+
Point(7) = {0, 5, -0, cl_v};
//+
Point(8) = {4, 0, -0, cl_v};
//+
Point(9) = {22, 0, -0, cl_v};

//+
Line(2) = {7, 6};
//+
Line(3) = {6, 4};
//+
Line(4) = {4, 8};
//+
Line(5) = {8, 9};
//+
Line(6) = {9, 2};
//+
Line(7) = {2, 7};
//+
Curve Loop(1) = {7, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};

Point{2} In Surface{1};
Point{4} In Surface{1};
Point{6} In Surface{1};
Point{7} In Surface{1};
Point{8} In Surface{1};
Point{9} In Surface{1};