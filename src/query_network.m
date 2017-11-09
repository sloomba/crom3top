function [ output, network ] = query_network( network, pattern, exact )
%QUERY_NETWORK
%Queries the given network (which has attributes adjmat and nodes) for the
%given pattern, and outputs the corresponding match and its in/outnodes.
if nargin<3
    exact = true;
end
if ~iscell(pattern)
    pattern = {pattern};
end
pattern = upper(pattern);
network.nodes = upper(network.nodes);
output = {};
if ~isfield(network, 'outranks')
    outranks = tiedrank(sum(network.adjmat, 2)');
    network.outranks = max(outranks)+1 - outranks;
end
if ~isfield(network, 'inranks')
    inranks = tiedrank(sum(network.adjmat, 1));
    network.inranks = max(inranks)+1 - inranks;
end
if ~isfield(network, 'pgr_undirected')
    network.pgr_undirected = pagerank(network.adjmat, 0.85, 0.00000001, true);
    pgr_undirected_ranks = tiedrank(network.pgr_undirected);
    network.pgr_undirected_ranks = max(pgr_undirected_ranks)+1 - pgr_undirected_ranks;
end
if ~isfield(network, 'pgr_directed')
    network.pgr_directed = pagerank(network.adjmat, 0.85, 0.00000001, false);
    pgr_directed_ranks = tiedrank(network.pgr_directed);
    network.pgr_directed_ranks = max(pgr_directed_ranks)+1 - pgr_directed_ranks;
end
if ~isfield(network, 'pgr_reversed')
    network.pgr_reversed = pagerank(network.adjmat', 0.85, 0.00000001, false);
    pgr_reversed_ranks = tiedrank(network.pgr_reversed);
    network.pgr_reversed_ranks = max(pgr_reversed_ranks)+1 - pgr_reversed_ranks;
end
for i=1:length(pattern)
    if exact
        idxs = find(strcmp(network.nodes, pattern{i}));
    else
        idxs = find(~cellfun(@isempty, strfind(network.nodes, pattern{i})));
    end
    matches = network.nodes(idxs);
    outnodes = network.adjmat(idxs,:);
    out = cell(length(idxs),1);
    for j=1:size(outnodes,1)
        out{j} = strjoin(network.nodes(logical(outnodes(j,:))));
    end
    innodes = network.adjmat(:,idxs);
    in = cell(length(idxs),1);
    for j=1:size(innodes,2)
        in{j} = strjoin(network.nodes(logical(innodes(:,j))));
    end
    output(i).pattern = pattern{i};
    output(i).matches = strjoin(matches, ' & ');
    output(i).outnodes = strjoin(out, ' & ');
    output(i).innodes = strjoin(in, ' & ');
    output(i).outdegree = mean(sum(outnodes, 2));
    output(i).outrank = mean(network.outranks(idxs));
    output(i).indegree = mean(sum(innodes, 1));
    output(i).inranks = mean(network.inranks(idxs));
    output(i).pagerank_undirected = mean(network.pgr_undirected(idxs));
    output(i).pagerank_undirected_ranks = mean(network.pgr_undirected_ranks(idxs));
    output(i).pagerank_directed = mean(network.pgr_directed(idxs));
    output(i).pagerank_directed_ranks = mean(network.pgr_directed_ranks(idxs));
    output(i).pagerank_reversed = mean(network.pgr_reversed(idxs));
    output(i).pagerank_reversed_ranks = mean(network.pgr_reversed_ranks(idxs));
end
output = struct2table(output);
end