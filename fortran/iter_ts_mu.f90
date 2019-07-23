subroutine iter_ts_mu(k, fofix, condfix, ts)

  implicit none

  ! INPUT VARIABLES

  real, intent (in) :: k, fofix, condfix

  ! OUTPUT VARIABLES
  real, intent (out) :: ts

  ! LOCAL VARIABLES
  integer :: niter = 0
  logical :: keepiterating
  real, parameter :: esice = 5.67e-5
  real, parameter :: tffresh = 273.16
  real, parameter :: tiny = 1e-6
  real :: ts_kelv, iru, cond, df_dt, dt, f

  ts = -20.
  keepiterating = .true.

  do while (keepiterating)
     ts_kelv = ts + tffresh
     iru = esice*ts_kelv**4
     cond = condfix - k*ts
     df_dt = -k-4*esice*ts_kelv**3
     f = fofix - iru + cond

     if (abs(df_dt) < tiny) then
        keepiterating = .false.
     else
        dt = -f/df_dt
        ts = ts + dt
        niter = niter + 1
        if (niter > 20 .or. abs(dt) > 0.001) then
           keepiterating = .false.
        end if
     end if
  end do

end subroutine iter_ts_mu
     
     
  
  
  
