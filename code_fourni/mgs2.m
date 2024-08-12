function Q = mgs2(A,B)

[~, n] = size(A);
[~, m] = size(B);
Q = B;

for i = 1:m
    for j = 1:n-1
        h = A(:,j)'*Q(:,i);
        Q(:,i) = Q(:,i) - h*A(:,j);
    end
    Q(:,i) = Q(:,i)/norm(Q(:,i));
end


end
