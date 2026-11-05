//+
cube = 0.03;
channel = 0.06;
Point(1) = {0, 0, 0, channel};
//+
Point(2) = {2.2, 0.4, 0, channel};
//+
Point(3) = {0.15, 0.15, 0, cube};
//+
Point(4) = {0.25, 0.25, 0, cube};
//+
Point(5) = {0.15, 0.25, 0, cube};
//+
Point(6) = {0.25, 0.15, 0, cube};
//+
Point(7) = {0., 0.4, 0, channel};
//+
Point(8) = {2.2, 0., 0, channel};
//+
Line(2) = {5, 3};
//+
Line(3) = {3, 6};
//+
Line(4) = {6, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {1, 8};
//+
Line(7) = {8, 2};
//+
Line(8) = {2, 7};
//+
Line(9) = {7, 1};
//+
Curve Loop(1) = {8, 9, 6, 7};
//+
Curve Loop(2) = {5, 2, 3, 4};
//+
Surface(1) = {1, 2};
