"""
Updates dictionary with a *subspace* version of cheap quadratic descent

The only difference between this and ./sc_cheap_quadratic_descent.py is that
here we add a regularization component to the loss function (and thus the
update) which penalizes the *alignment of dictionary elements that belong
to the same subgroup*. This encourages elements of the same subgroup to be
different, otherwise dictionary learning can some times generate duplicate
elements.
"""
import torch

def run(images, dictionary, codes, group_assignments,
        hessian_diagonal, alignment_penalty, stepsize=0.001, num_iters=1,
        lowest_code_val=0.001, normalize_dictionary=True):
  """
  Runs num_iters steps of an approximate quadratic descent w/ regularization

  Parameters
  ----------
  images : torch.Tensor(float32, size=(b, n))
      An array of images (probably just small patches) that we want to find the
      sparse code for. n is the size of each image and b is the number of
      images in this batch
  dictionary : torch.Tensor(float32, size=(s, n))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  codes : torch.Tensor(float32, size=(b, s))
      This is the current set of codes for a batch of images. s is the
      dimensionality of the code and b is the number of images in the batch
  group_assignments : list(array_like)
      Elements of this list identify which dictionary elements belong to
      a particular group. Our convention is the following: Suppose we have
        group_assignments = [[0, 2, 5], [1], [2, 3, 4, 5]]
      This specifies three groups. group 0 is comprised of elements 0, 2,
      and 5 from the dictioanary, group 1 is composed of element 1, and
      group 2 is composed of elements 2, 3, 4, and 5. Notice that each group
      can be of a different size and elements of the dictionary can
      participate in multiple groups. This is used to compute a regularization
      penalty that applies *within subgroups*
  hessian_diagonal : torch.Tensor(float32, size=(s,))
      An estimate of the diagonal of the hessian that we'll compute outside of
      this function call.
  alignment_penalty : float
      We impose a regularization penalty on the alignment of dictionary
      elements that are part of the same group. This is the lagrange
      multiplier beta, which weights this penalty in the full loss function.
  stepsize : torch.Tensor(float32)
      The step size for each iteration of the quad. descent. Keep this small.
      Default 0.001.
  num_iters : int, optional
      Number of steps of quad. descent to run. Default 1.
  lowest_code_val : float, optional
      Used to condition the hessian diagonal to not be too small. Default 0.001
  normalize_dictionary : bool, optional
      If true, we normalize each dictionary element to have l2 norm equal to 1
      before we return. Default True.
  """
  if alignment_penalty != 0:
    accum_regularization_gradients = dictionary.new_zeros(dictionary.size())
    for iter_idx in range(num_iters):
      # compute regularization gradients first. We accumulate gradients from
      # multiple groups, if necessary.
      if iter_idx > 0:
        accum_regularization_gradients.mul_(0.)
      for g_idx in range(len(group_assignments)):
        accum_regularization_gradients[group_assignments[g_idx]] = (
            accum_regularization_gradients[group_assignments[g_idx]] +
            regularization_gradients(dictionary[group_assignments[g_idx]],
                                     normalize_dictionary))
      dict_update = stepsize * ((torch.mm(
        codes.t(), torch.mm(codes, dictionary) - images) / codes.size(0)) +
        alignment_penalty * accum_regularization_gradients)
      # ^first term is the gradient of the reconstruction error, the second term
      #  is the (already computed) gradient of the alignment regularization
      dict_update.div_(hessian_diagonal[:, None] + lowest_code_val)
      dictionary.sub_(dict_update)
      if normalize_dictionary:
        dictionary.div_(dictionary.norm(p=2, dim=1)[:, None])
  else:
    # just vanilla sc_cheap_quadratic_descent, no ext. bookkeeping (and faster)
    for iter_idx in range(num_iters):
      dict_update = stepsize * (torch.mm(
        codes.t(), torch.mm(codes, dictionary) - images) / codes.size(0))
      dict_update.div_(hessian_diagonal[:, None] + lowest_code_val)
      dictionary.sub_(dict_update)
      if normalize_dictionary:
        dictionary.div_(dictionary.norm(p=2, dim=1)[:, None])


def regularization_gradients(dictionary, dict_is_normalized):
  """
  Computes the gradients of the regularization term within a dictionary

  Our regularization penalty typically applies to subsets of a larger
  dictionary--the elements that belong to the same "group". Therefore,
  the dictionary we pass in here will likely have between 2 and 8
  elements (rows), but we could in theory compute it on an aribitrarily large
  dictionary.
  """
  # cos(\phi_i, \phi_j) has a gradient with two terms -- one that depends on
  # \phi_i, which I call axis_0_term below, and one that depends on \phi_j,
  # which I call axis_1_term. we subtract axis_0_term from axis_1_term and
  # modulate by the sign of the cosine similarity. This produces the gradient
  # of the unsigned cosine similarity between every pair of dictionary elements
  if dict_is_normalized:
    cos_sims = torch.mm(dictionary, dictionary.t())[:, :, None]
    axis_0_term = (cos_sims *
        torch.cat(dictionary.size(0)*[dictionary[:, None, :]], dim=1))
    axis_1_term = torch.cat(dictionary.size(0)*[dictionary[None, :, :]], dim=0)
  else:
    # The more general expression accounts for the norms
    norms = torch.norm(dictionary, p=2, dim=1, keepdim=True)
    cos_sims = (torch.mm(dictionary, dictionary.t()) /
                torch.mm(norms, norms.t()))[:, :, None]
    axis_0_term = ((cos_sims / (norms**2)[:, None]) *
         torch.cat(dictionary.size(0)*[dictionary[:, None, :]], dim=1))
    axis_1_term = (
        torch.cat(dictionary.size(0)*[dictionary[None, :, :]], dim=0) /
        torch.mm(norms, norms.t())[:, :, None])
  tensor_of_gradients = torch.sign(cos_sims) * (axis_1_term - axis_0_term)
  # the gradient for each dictionary element is the sum along axis 1 of this
  # tensor -- it gives a gradient contribution for each of the unsigned cosine
  # similarities to the *other* vectors. The diagonal will be zero (by the
  # definition of the gradient) so we don't need to do any fancy masking of
  # the diagonal.
  return torch.sum(tensor_of_gradients, dim=1)
