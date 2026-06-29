function J = debug_objective_wrapper(fun, x, node_id, iteration)

    fprintf('\n--- Debug objective ---\n');
    fprintf('ADMM iteration: %d, node: %d\n', iteration, node_id);
    fprintf('x = [%g, %g, %g, %g]\n', x(1), x(2), x(3), x(4));

    if any(~isfinite(x))
        error('x contains NaN or Inf at node %d, iteration %d.', node_id, iteration);
    end

    if ~isreal(x)
        error('x is complex at node %d, iteration %d.', node_id, iteration);
    end

    try
        J = fun(x);
    catch ME
        fprintf('Objective crashed at node %d, iteration %d.\n', node_id, iteration);
        fprintf('x = \n');
        disp(x);
        rethrow(ME);
    end

    fprintf('J = %g\n', J);

    if isempty(J)
        error('Objective returns empty value at node %d, iteration %d.', node_id, iteration);
    end

    if ~isscalar(J)
        error('Objective returns non-scalar value at node %d, iteration %d.', node_id, iteration);
    end

    if ~isreal(J)
        fprintf('Complex objective detected.\n');
        fprintf('real(J) = %.16e\n', real(J));
        fprintf('imag(J) = %.16e\n', imag(J));
        error('Objective returns complex value at node %d, iteration %d.', ...
              node_id, iteration);
    end

    if ~isfinite(J)
        error('Objective returns NaN or Inf at node %d, iteration %d. J = %g', ...
              node_id, iteration, J);
    end
end