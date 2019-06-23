#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

//此类中的所有成员函数都是ceres::localParameterization类中的纯虚函数
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    //因为位姿只有六维，而四元数却有七位，具有冗余,GlobalSize表示的是四元数表示法的维度，localSize表示的是实际位姿具有的维度
    virtual int GlobalSize() const { return 7; };//The dimension of the ambient space in which the parameter block x lives.
    virtual int LocalSize() const { return 6; };//The size of the tangent space that Δx lives in.
};
