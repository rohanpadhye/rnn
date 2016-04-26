-- Adapted from https://gist.github.com/BlackBulletIV/4084042

local LinkedList = torch.class("nn.LinkedList")
local ListNode = torch.class("nn.ListNode")

-- Initialize a new linked list node with some data
function ListNode:__init(data, list)
   self._data = data
   self._list = list
   self._prev = nil
   self._next = nil
end

-- Initialize an empty linked list
function LinkedList:__init()
   self.first = nil
   self.last = nil
   self.length = 0
end

-- Converts a value to a linked list node
function LinkedList:getNode(x)
   return nn.ListNode(x, self)
   --if x._list == self then
   --   return x
   --else
   --   return ListNode(x, self)
   --end
end

-- Add an item at the end of the list
function LinkedList:pushBack(x)
   local t = self:getNode(x)
   -- check if there are some nodes in the list
   if self.last then
      self.last._next = t
      t._prev = self.last
      self.last = t
   else
      -- this is the only node
      self.first = t
      self.last = t
   end
  
   -- increment length and return reference to node
   self.length = self.length + 1
   return t
end

-- Add an item at the beginning of the list
function LinkedList:pushFront(x)
   local t = self:getNode(x)
   -- check if there are some nodes in the list
   if self.first then
      self.first._prev = t
      t._next = self.first
      self.first = t
   else
      -- this is the only node
      self.first = t
      self.last = t
   end

   -- increment length and return reference to node
   self.length = self.length + 1
   return t
end

-- Remove an item from the end of the list
function LinkedList:popBack()
   -- empty list results in a nil value
   if not self.last then return end
   -- otherwise the return value is the last node
   local ret = self.last
   -- disassociate the node from the list
   ret._list = nil
  
   -- if a previous node existed, remove the connection
   if ret._prev then
      ret._prev._next = nil
      self.last = ret._prev
      ret._prev = nil
   else
      -- this was the only node
      self.first = nil
      self.last = nil
   end
  
   -- decrement length and return value
   self.length = self.length - 1
   return ret._data
end

-- Remove an item from the beginning of the list
function LinkedList:popFront()
   -- empty list results in a nil value
   if not self.last then return end
   -- otherwise the return value is the first node
   local ret = self.first
   -- disassociate the node from the list
   ret._list = nil
  
   -- if a next node existed, remove the connection
   if ret._next then
      ret._next._prev = nil
      self.first = ret._next
      ret._next = nil
   else
      self.first = nil
      self.last = nil
   end
  
   -- decrement length and return value
   self.length = self.length - 1
   return ret._data
end


-- Utility for iterator 
local function iterate(self, current)
  if not current then
    current = self.first
  elseif current then
    current = current._next
  end
  
  return current
end

-- Iterator for use in for-loops
function LinkedList:iterate()
  return iterate, self, nil
end

-- Returns whether a given node is contained in the list
function LinkedList:contains(t)
   return t._list == self
end

-- Removes a node from a list
function LinkedList:removeNode(t)
   -- ensure that node being removed is actually part of the list
   assert(self:contains(t))

   -- remove next/previous connections if they exist
   if t._next then
      if t._prev then
         t._next._prev = t._prev
         t._prev._next = t._next
      else
         -- this was the first node
         t._next._prev = nil
         self.first = t._next
      end
   elseif t._prev then
      -- this was the last node
      t._prev._next = nil
      self.last = t._prev
   else
      -- this was the only node
      self.first = nil
      self.last = nil
   end

   -- clear node associations
   t._next = nil
   t._prev = nil
   t._list = nil

   -- decrement length
   self.length = self.length - 1
end

-- Brings a node to the front, if it is not already
function LinkedList:bringToFront(t)
   if (self.first == t) then
      return
   end

   assert(self:contains(t))
   assert(t._prev)

   -- remove next/previous connections if they exist
   t._prev._next = t._next
   if t._next then
      t._next._prev = t._prev
   else
      self.last = t._prev
   end
   
   -- make this the first node
   self.first._prev = t
   t._prev = nil
   t._next = self.first
   self.first = t

end

function LinkedList:__tostring()
   local str = "  List: "
   for elem in self:iterate() do
      str = str .. elem._data .. " "
   end
   str = str .. " (#" .. self.length .. ")"
   return str
end

