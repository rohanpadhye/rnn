------------------------------------------------------------------------
--[[ ConvertingSequencer ]]--
-- Encapsulates a Module of a type different than the sequencer.
-- This is useful for example when the recurrent module is of type
-- torch.CudaTensor but the sequence lengths are too long for the entire
-- sequence to remain in GPU memory. In such a case, the ConvertingSequencer
-- can be set to type torch.FloatTensor and it will copy data to and from
-- the GPU at each time-step without loss of precision. This effectively
-- makes the GPU memory a cache, with the system memory as the backing
-- store.
--
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
------------------------------------------------------------------------
assert(not nn.ConvertingSequencer, "update nnx package : luarocks install nnx")
local ConvertingSequencer, parent = torch.class('nn.ConvertingSequencer', 'nn.Sequencer')

function ConvertingSequencer:__init(module)
   parent.__init(self, module)
   -- self._type should ideally be initialized by nn.Module
   if not self._type then
      self._type = torch.Tensor():type()
   end
end

function ConvertingSequencer:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "expecting input table")

   -- Note that the Sequencer hijacks the rho attribute of the rnn
   self.module:maxBPTTstep(#inputTable)
   if self.train ~= false then -- training
      if not (self._remember == 'train' or self._remember == 'both') then
         self.module:forget()
      end
      self.output = {}
      for step, input in ipairs(inputTable) do
         -- convert input from sequencer type to recurrent module type
         input = nn.rnn.recursiveType(input, self.module:type())
         -- compute output 
         local output = self.module:updateOutput(input)
         -- convert output from recurrent module type to sequencer type
         self.output[step] = nn.rnn.recursiveType(output, self:type())
      end
   else -- evaluation
      if not (self._remember == 'eval' or self._remember == 'both') then
         self.module:forget()
      end
      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table
      for step, input in ipairs(inputTable) do
         -- convert input from sequencer type to recurrent module type
         input = nn.rnn.recursiveType(input, self.module:type())
         -- compute output 
         local output = self.module:updateOutput(input)
         -- convert output from recurrent module type to sequencer type (force copy)
         self.output[step] = nn.rnn.recursiveTypedCopy(
            self.output[step] or table.remove(self._output, 1),
            output, self:type())
      end
      -- remove extra output tensors (save for later)
      for i=#inputTable+1,#self.output do
         table.insert(self._output, self.output[i])
         self.output[i] = nil
      end
   end
   
   return self.output
end

function ConvertingSequencer:updateGradInput(inputTable, gradOutputTable)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   self.gradInput = {}
   for step=#gradOutputTable,1,-1 do
      -- convert input from sequencer type to recurrent module type
      local input = nn.rnn.recursiveType(inputTable[step], self.module:type())
      -- convert gradOutput from sequencer type to recurreent module type
      local gradOutput = nn.rnn.recursiveType(gradOutputTable[step], self.module:type())
      -- update gradInput
      local gradInput = self.module:updateGradInput(input, gradOutput)
      -- convert gradInput from recurrent module type to sequencer type
      self.gradInput[step] = nn.rnn.recursiveType(gradInput, self:type())
   end
   
   assert(#inputTable == #self.gradInput, #inputTable.." ~= "..#self.gradInput)

   return self.gradInput
end

function ConvertingSequencer:accGradParameters(inputTable, gradOutputTable, scale)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   for step=#gradOutputTable,1,-1 do
      -- convert input from sequencer type to recurrent module type
      local input = nn.rnn.recursiveType(inputTable[step], self.module:type())
      -- convert gradOutput from sequencer type to recurreent module type
      local gradOutput = nn.rnn.recursiveType(gradOutputTable[step], self.module:type())
      -- accumulate gradParameters
      self.module:accGradParameters(input, gradOutput, scale)
   end   
end

function ConvertingSequencer:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
   assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")
   
   -- back-propagate through time (BPTT)
   for step=#gradOutputTable,1,-1 do
      -- convert input from sequencer type to recurrent module type
      local input = nn.rnn.recursiveType(inputTable[step], self.module:type())
      -- convert gradOutput from sequencer type to recurreent module type
      local gradOutput = nn.rnn.recursiveType(gradOutputTable[step], self.module:type())
      -- accumulate and update gradParameters
      self.module:accUpdateGradParameters(input, gradOutput, lr)
   end     
end

-- Override Module.type() to prevent contained modules
-- from being type-casted.
function ConvertingSequencer:type(type, tensorCache)
   if not type then
      return self._type
   else
      self._type = type
      return self
   end
end

