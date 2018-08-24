for nup=0,7 do
    for ndown=0,7 do
        if nup == 0 and ndown == 0 then
        else
            L = 2.0309825951265186
            rss = (3*L^3/(4*3.141592653589793*(nup+ndown)))^(1.0/3.0)
            sys = ueg {
                nel = nup + ndown,
                ms = nup - ndown,
                sym = 1,
                dim = 3,
                cutoff = 0.5,
                rs = rss,
            }

            fci {
                sys = sys,
                fci = {
                },
            }
        end
    end
end
