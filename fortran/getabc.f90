subroutine getabc(ti, tbot, zeta, delta, k, eta, ni, lfirst, a, b, c, r)
  implicit none

  ! INPUT VARIABLES
  real, dimension(:), intent (in) :: ti, zeta, delta, k, eta
  integer, intent (in) :: ni, lfirst
  real, intent(in) :: tbot
  ! OUTPUT VARIABLES
  real, dimension(ni+2), intent (out) :: a, b, c, r

  ! LOCAL variables

  real :: alph, bet
  integer, dimension (ni-lfirst) :: layers
  integer :: i

  alph = 3.
  bet = -1./3.

  layers = (/ (i, i=lfirst,ni-1) /)

  a(layers+2) = -eta(layers+1)*k(layers+1)
  c(layers+2) = -eta(layers+1)*k(layers+2)
  b(layers+2) = 1-c(layers+2)-a(layers+2)
  r(layers+2) = -zeta(layers+1)+a(layers+2)*ti(layers)+b(layers+2)*ti(layers+1)+c(layers+2)*ti(layers+2)

  a(ni+2) = -eta(ni+1)*(k(ni+1)-bet*k(ni+2))
  c(ni+2) = 0.
  b(ni+2) = 1+eta(ni+1)*(k(ni+1)+alph*k(ni+2))
  r(ni+2) = -zeta(ni+1)+a(ni+2)*ti(ni)+b(ni+2)*ti(ni+1)-eta(ni+1)*(alph+bet)*k(ni+2)*tbot

end subroutine getabc
