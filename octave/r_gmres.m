fprintf("size,time,iter,resNorm,precond,round,sam,run\n");

for run=1:5
    for i=1:3
        [x, r, r2] = gmresTest(10^i, true, true);
        for j=1:size(r)
            fprintf("%d,%f,%d,%f,1,1,1,%d\n", 10^i, x, j, r(j), run);
        end
        
        for j=1:size(r2)
            fprintf("%d,%f,%d,%f,1,2,1,%d\n", 10^i, x, j, r2(j), run);
        end

    end
end

for run=1:5
    for i=1:2
        [x, r, r2] = gmresTest(10^i, false);
        for j=1:size(r)
            fprintf("%d,%f,%d,%f,0,1,0,%d\n", 10^i, x, j, r(j), run);
        end
        for j=1:size(r2)
            fprintf("%d,%f,%d,%f,0,2,0,%d\n", 10^i, x, j, r2(j), run);
        end

    end
end

for run=1:5
    for i=1:4
        [x, r, r2] = gmresTest(10^i, true);
        for j=1:size(r)
            fprintf("%d,%f,%d,%f,1,1,0,%d\n", 10^i, x, j, r(j), run);
        end
        
        for j=1:size(r2)
            fprintf("%d,%f,%d,%f,1,2,0,%d\n", 10^i, x, j, r2(j), run);
        end

    end
end


