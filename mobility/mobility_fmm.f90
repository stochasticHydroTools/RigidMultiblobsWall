!
! This is an interface to call the library stfmmlib3d 
! from python
!


! This function computes the Stokes interaction
! in the presence of a no-slip wall at z=0.
! Particles should be located at z>0.
!subroutine fmm_stokeslet_half(ier, iprec, N, N3, X, F, U, RH, viscosity)
!subroutine fmm_stokeslet_half(ier, iprec, N, N3, x)
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
  
  !call stfmm3dpartself (ier, iprec, N, source, 1, sigma_sl, 0, sigma_dl, sigma_dv, 1, pot, pre, 0, grad)
  call sthfmm3dpartself(ier, iprec, itype, N, x, 1, F, 0, sigma_dl,sigma_dv, 1, u, pre, 0, grad)
    
  
  
  ! Scale velocity and add self mobility
  do i=1, N
     U(1,i) = U(1,i) / (4 * pi * viscosity) + F(1,i) / (6 * pi * viscosity * RH)
     U(2,i) = U(2,i) / (4 * pi * viscosity) + F(2,i) / (6 * pi * viscosity * RH)
     U(3,i) = U(3,i) / (4 * pi * viscosity) + F(3,i) / (6 * pi * viscosity * RH)
  end do


  write(*,*) 'fmm_Stokeslet_half DONE'
end subroutine fmm_stokeslet_half



program test
  implicit none


  write(*,*) "# DONE"
end program test
