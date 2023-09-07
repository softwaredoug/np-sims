#ifndef HEAP_H
#define HEAP_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef UINT64_MAX
#define UINT64_MAX 0xffffffffffffffff
#endif

struct Heap {
  uint64_t *score_arr;
  uint64_t *idx_arr;
  uint64_t size;
  uint64_t capacity;
};

static void swap(uint64_t *a, uint64_t *b) {
  uint64_t t = *a;
  *a = *b;
  *b = t;
}

static void init_heap(struct Heap* heap, uint64_t* idx_arr, uint64_t* score_arr, uint64_t capacity) {
  heap->size = 0;
  heap->capacity = capacity;
  heap->score_arr = score_arr;
  heap->idx_arr = idx_arr;
  memset(heap->score_arr, -1, capacity * sizeof(uint64_t));
  memset(heap->idx_arr, -1, capacity * sizeof(uint64_t));
}

static void free_heap(struct Heap* heap) {
  free(heap->score_arr);
}

static void heapify(struct Heap *heap, uint64_t i) {
  int largest = i;

  int l = 2*i + 1;
  int r = 2*i + 2;

  if (l < heap->size && heap->score_arr[l] > heap->score_arr[largest]) {
    largest = l;
  }

  if (r < heap->size && heap->score_arr[r] > heap->score_arr[largest]) {
    largest = r;
  }

  if (largest != i) {
    swap(&heap->score_arr[i], &heap->score_arr[largest]);
    swap(&heap->idx_arr[i], &heap->idx_arr[largest]);

    heapify(heap, largest);
  }
}

static void heap_insert(struct Heap* heap, uint64_t idx, uint64_t score) {
  if (heap->size < heap->capacity) {
    heap->score_arr[heap->size] = score;
    heap->idx_arr[heap->size] = idx;
    heap->size++;
    if (heap->size == heap->capacity) {
      heapify(heap, 0);
    }
  }
  else if (score < heap->score_arr[0]) {
    heap->score_arr[0] = score;
    heap->idx_arr[0] = idx;
    heapify(heap, 0);
  }
}

#endif
