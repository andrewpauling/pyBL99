subroutine tridag(a, b, c, r, n, n1, u)

  implicit none

  ! Input variables 
  real, dimension (:), intent(in) :: a, b, c, r
  integer, intent(in) :: n, n1

  ! Output variables
  real, dimension (n), intent(out) :: u

  ! Local variables
  real, dimension (n1+2) :: gam
  real :: bet
  integer :: layer

  bet = b(1)
  u(1) = r(1)/bet

  do layer=2,n
     gam(layer) = c(layer-1)/bet
     bet = b(layer) -a(layer)*gam(layer)

     u(layer) = (r(layer) - a(layer)*u(layer-1))/bet

  end do

  do layer=n-1,1,-1
     u(layer) = u(layer) -gam(layer+1)*u(layer+1)
  end do

end subroutine tridag

  
