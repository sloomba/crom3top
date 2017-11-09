function [ output ] = query_regulation( gene_names, up_regulated, down_regulated, diagnostic_table, exact )
%QUERY_REGULATION, given the diagnostic table which is the output of QUERY
%TABLE, and a list of gene names and list of up/down regulated gene indexes,
%outputs a structure containing the up/down list to be used for PROM. You
%must take an intersection with the genes present in the metabolicmodel
%before giving to PROM.
if nargin<5
    exact = true;
end
pattern = {};
for i=1:height(diagnostic_table)
    if diagnostic_table.outdegree(i)>0
        pattern = cat(2, pattern, strsplit(diagnostic_table.outnodes{i}, {' ','&'}));
    else
        pattern = cat(2, pattern, diagnostic_table.pattern{i});
    end
end
gene_names = upper(gene_names);
pattern = upper(pattern);
output = struct('up',[],'down',[]);
for i=1:length(pattern)
    if exact
        idxs = find(strcmp(gene_names, pattern{i}));
    else
        idxs = find(~cellfun(@isempty, strfind(gene_names, pattern{i})));
    end
    if all(down_regulated(idxs))
        output.down{length(output.down)+1} = pattern{i};
    else
        if all(up_regulated(idxs))
            output.up{length(output.up)+1} = pattern{i};
        end
    end
end
output = struct2table(output);
end