!
! This is an interface to call the library stfmmlib3d 
! from python
!


! This function computes the Stokes interaction
! in the presence of a no-slip wall at z=0.
! Particles should be located at z>0.
subroutine fmm_stokeslet_half(x, F, U, ier, iprec, RH, viscosity, N)
  implicit none
  integer, intent(in) :: iprec, N
  integer, intent(inout) :: ier
  real*8, intent(in) :: RH, viscosity
  real*8, intent(in) :: X(3,N), F(3,N)
  real*8, intent(inout) :: U(3, N)
  real*8 sigma_dl(3,N), sigma_dv(3,N), pre(N), grad(3,3,N)
  real*8 :: pi
  integer :: itype, i
  itype = 2
  pi = 4.0d0 * datan(1.0d0)
  
  call sthfmm3dpartself(ier, iprec, itype, N, x, 1, F, 0, sigma_dl,sigma_dv, 1, u, pre, 0, grad)
  
  ! Scale velocity and add self mobility
  do i=1, N
     U(1,i) = U(1,i) / (4 * pi * viscosity) + F(1,i) / (6 * pi * viscosity * RH)
     U(2,i) = U(2,i) / (4 * pi * viscosity) + F(2,i) / (6 * pi * viscosity * RH)
     U(3,i) = U(3,i) / (4 * pi * viscosity) + F(3,i) / (6 * pi * viscosity * RH)
  end do
end subroutine fmm_stokeslet_half


! This function computes the Stokes interaction
! with the Rotne-Prager tensor
subroutine fmm_rpy(x, F, U, ier, iprec, RH, viscosity, N)
  implicit none
  integer, intent(in) :: iprec, N
  integer, intent(inout) :: ier
  real*8, intent(in) :: RH, viscosity
  real*8, intent(in) :: X(3,N), F(3,N)
  real*8, intent(inout) :: U(3, N)
  real*8 sigma_dl(3,N), sigma_dv(3,N), pre(N), grad(3,3,N)
  real*8 targetLocations(3,0), POTtarg(3,0), PREtarg(0), GRADtarg(3,3,0)
  real*8 :: pi
  integer :: i
  pi = 4.0d0 * datan(1.0d0)
  
  call rpyfmm3dparttarg(ier, iprec, N, x, 1, F, 0, sigma_dl, sigma_dv, 1, U, & 
       pre, 0, grad, 0, targetLocations, 0, POTtarg, PREtarg, 0, GRADtarg, RH, 0)  
  
  ! Scale velocity 
  do i=1, N
     U(1,i) = U(1,i) / viscosity
     U(2,i) = U(2,i) / viscosity
     U(3,i) = U(3,i) / viscosity
  end do
end subroutine fmm_rpy

