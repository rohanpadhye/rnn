
function rnn.recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = rnn.recursiveResizeAs(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function rnn.recursiveSet(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = rnn.recursiveSet(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:set(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function rnn.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = rnn.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function rnn.recursiveTypedCopy(t1, t2, type_str)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = rnn.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      if torch.isTensor(t1) then
         if t1:type() ~= type_str then
            error("expecting destination tensor of type " .. type_str ..
               " but got " .. torch.type(t1) .. " instead")
         end
      else
         t1 = torch.Tensor():type(type_str)
      end
      t1:resize(t2:size()):copy(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function rnn.recursiveAdd(t1, t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = rnn.recursiveAdd(t1[key], t2[key])
      end
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      t1:add(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function rnn.recursiveTensorEq(t1, t2)
   if torch.type(t2) == 'table' then
      local isEqual = true
      if torch.type(t1) ~= 'table' then
         return false
      end
      for key,_ in pairs(t2) do
          isEqual = isEqual and rnn.recursiveTensorEq(t1[key], t2[key])
      end
      return isEqual
   elseif torch.isTensor(t1) and torch.isTensor(t2) then
      local diff = t1-t2
      local err = diff:abs():max()
      return err < 0.00001
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
end

function rnn.recursiveNormal(t2)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = rnn.recursiveNormal(t2[key])
      end
   elseif torch.isTensor(t2) then
      t2:normal()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function rnn.recursiveFill(t2, val)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = rnn.recursiveFill(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      t2:fill(val)
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function rnn.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for i = 1, #param do
         param[i] = rnn.recursiveType(param[i], type_str)
      end
   else
      if torch.typename(param) and 
        torch.typename(param):find('torch%..+Tensor') then
         param = param:type(type_str)
      end
   end
   return param
end

function rnn.recursiveSum(t2)
   local sum = 0
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         sum = sum + rnn.recursiveSum(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      return t2:sum()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return sum
end

function rnn.recursiveNew(t2)
   if torch.type(t2) == 'table' then
      local t1 = {}
      for key,_ in pairs(t2) do
         t1[key] = rnn.recursiveNew(t2[key])
      end
      return t1
   elseif torch.isTensor(t2) then
      return t2.new()
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
end

function rnn.stepCloneRecursiveType(param, type, tensorCache)
   tensorCache = tensorCache or {}

   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = rnn.stepCloneRecursiveType(v, type, tensorCache)
      end
   elseif torch.isTypeOf(param, 'nn.Module') or
          torch.isTypeOf(param, 'nn.Criterion') then
      -- recurrent modules will handle their own step clones
      if not torch.isTypeOf(param, 'nn.AbstractRecurrent') then
         for k, v in pairs(param) do
            param[k] = rnn.stepCloneRecursiveType(v, type, tensorCache)
         end
      end
   elseif torch.isTensor(param) then
      if torch.typename(param) ~= type then
         local newparam
         if tensorCache[param] then
            newparam = tensorCache[param]
         else
            -- Use the default recursiveType implementation to cast tensor type
            -- and update the tensorCache
            newparam = nn.utils.recursiveType(param, type, tensorCache) 
         end
         param = newparam
      end
   end
   return param
end
