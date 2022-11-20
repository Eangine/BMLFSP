function [batchCell] = gengerateMiniBatch(spMatrix)
% 0 generate mini-batch 
    rand('state',sum(100*clock));
    batchSize = 16;
    ParticipateSampleIndex = find(spMatrix(:)~=0);
    numOfSample = length(find(spMatrix(:)~=0));
    batchIdx = randperm(numOfSample);
    numOfBatch = floor(numOfSample/batchSize);
    if numOfSample/batchSize > floor(numOfSample/batchSize)
        numOfBatch = numOfBatch + 1;
        batchCell = cell(1,numOfBatch);
        for i = 1:numOfBatch-1
            batchCell{i} = ParticipateSampleIndex( batchIdx( (i-1)*batchSize+1:i*batchSize ) );
        end
        restNum = batchSize*numOfBatch - numOfSample;
        restOfIdx = randperm(numOfSample,restNum);
        batchCell{numOfBatch} = ParticipateSampleIndex( [batchIdx((numOfBatch-1)*batchSize+1:numOfSample) restOfIdx] );
    else
        batchCell = cell(1,numOfBatch);
        for i = 1:numOfBatch
            batchCell{i} = ParticipateSampleIndex( batchIdx( (i-1)*batchSize+1:i*batchSize ) );
        end
    end
end