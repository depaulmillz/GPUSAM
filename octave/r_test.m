n = 100;
x = rand(n);
y = rand(n);

tic
parfor i=1:n
    for j = 1:n
        x(i,j) = x(i,j) + y(i,j);
    end
end
toc

tic
for i=1:4:n
    for j = 1:4:n
        for ii=1:4
            if ii + i <= n
                for jj=1:4
                    x(i + ii,j + jj) = x(i + ii,j + jj) + y(i + ii,j + jj);
                    if j + jj > n
                        break
                    end
                end
            end
        end
    end
end
toc