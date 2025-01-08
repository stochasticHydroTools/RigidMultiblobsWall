function R = Rot_From_Q(s,p)
    P = [0, -1*p(3), p(2)
        p(3), 0, -1*p(1)
        -1*p(2), p(1), 0];
    R = 2*((p'*p) + (s^2-0.5)*eye(3) + s*P);
end