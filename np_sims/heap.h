#ifndef HEAP_H
#define HEAP_H

#include <stdint.h>

static void swap(uint64_t *a, uint64_t *b) {
  uint64_t t = *a;
  *a = *b;
  *b = t;
}

/* Maintain a max heap, put the largest element at the top */
static void heapify(uint64_t *arr, uint64_t *idx_arr, uint64_t n, uint64_t i) {

  int largest = i;

  int l = 2*i + 1;
  int r = 2*i + 2;

  if (l < n && arr[l] > arr[largest]) {
    largest = l;
  }

  if (r < n && arr[r] > arr[largest]) {
    largest = r;
  }

  if (largest != i) {
    swap(&arr[i], &arr[largest]);
    swap(&idx_arr[i], &idx_arr[largest]);

    heapify(arr, idx_arr, n, largest);
  }
}

/* Push only if the new value is smaller than the largest value in the heap */
static void maybe_push_heap(uint64_t *arr, uint64_t *idx_arr, uint64_t n, uint64_t new_val, uint64_t new_idx) {
  if (new_val > arr[0]) {
    return;
  }
  arr[0] = new_val;
  idx_arr[0] = new_idx;
  heapify(arr, idx_arr, n, 0);
}



#endif
